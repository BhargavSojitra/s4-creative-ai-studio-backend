import type { GoogleGenAI } from "@google/genai";
import {
  INPUT_IMAGE_FIELDS,
  type JobDoc,
  type JobPose,
  jobs,
  ObjectId,
} from "./db";
import {
  briefError,
  createGeminiClient,
  extractErrorMessage,
  type GenerateRequestBody,
  type GeneratedImage,
  generatePoseImage,
  isRetryableError,
  type PreparedSession,
  prepareSession,
} from "./gemini";
import { extensionForMimeType, getObject, objectKey, putObject } from "./storage";
import { backfillThumbnails, createThumbnail } from "./thumbnails";

// Number of poses generated at the same time across all jobs. Pro Image on
// Vertex starts returning 429s at a handful of concurrent calls, so keep this
// at 1-2 unless Provisioned Throughput is in place.
const CONCURRENCY = Math.max(1, Number(process.env.WORKER_CONCURRENCY) || 1);
const MAX_ATTEMPTS = Math.max(1, Number(process.env.WORKER_MAX_ATTEMPTS) || 3);
const POLL_MS = 2000;
const ACTIVE_STATUSES = ["queued", "running"] as const;

type Claim = { job: JobDoc; poseIndex: number };

// Per-job in-memory cache so the input images are pulled from R2 once, not
// once per pose. Evicted when the job reaches a terminal state.
type JobCache = {
  session?: Promise<PreparedSession>;
  reference?: Promise<GeneratedImage>;
};

const jobCache = new Map<string, JobCache>();
let ai: GoogleGenAI | null = null;
let activeSlots = 0;
let scheduling = false;
let started = false;

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const slug = (value: string): string =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "pose";

const backoffMs = (attempt: number): number => Math.pow(2, attempt) * 15000 + Math.random() * 5000;

const getAi = (): GoogleGenAI => {
  if (!ai) ai = createGeminiClient();
  return ai;
};

const cacheFor = (jobId: ObjectId): JobCache => {
  const key = jobId.toHexString();
  let entry = jobCache.get(key);
  if (!entry) {
    entry = {};
    jobCache.set(key, entry);
  }
  return entry;
};

const loadSession = async (job: JobDoc): Promise<PreparedSession> => {
  const body: Partial<GenerateRequestBody> = {
    ...job.params,
    selectedPoses: job.poses.map(({ id, title, prompt, category }) => ({ id, title, prompt, category })),
  };

  await Promise.all(
    INPUT_IMAGE_FIELDS.map(async (field) => {
      const ref = job.inputs[field];
      if (!ref) return;
      const buffer = await getObject(ref.key);
      body[field] = { data: buffer.toString("base64"), mimeType: ref.mimeType };
    }),
  );

  return prepareSession(body as GenerateRequestBody);
};

const loadReference = async (key: string, mimeType: string): Promise<GeneratedImage> => {
  const buffer = await getObject(key);
  return { data: buffer.toString("base64"), mimeType };
};

// Picks the pose index this job can run right now, or -1.
// Until the job has a reference image, only one pose may run (it becomes the
// reference). After that any pending pose can run in parallel.
const claimablePoseIndex = (job: JobDoc, now: Date): number => {
  const generating = job.poses.some((p) => p.status === "generating");
  if (!job.referenceKey && generating) return -1;

  return job.poses.findIndex(
    (p) => p.status === "pending" && (!p.nextAttemptAt || p.nextAttemptAt <= now),
  );
};

// Scheduling order: urgent first, then oldest. First pass hands each slot to a
// different job so two waiting users each get one; second pass lets both slots
// work the same job when nothing else is waiting.
const claimNextPose = async (): Promise<Claim | null> => {
  const now = new Date();
  const candidates = await jobs()
    .find({ status: { $in: [...ACTIVE_STATUSES] } })
    .sort({ priority: -1, createdAt: 1 })
    .toArray();

  for (const allowSharedJob of [false, true]) {
    for (const job of candidates) {
      const generatingCount = job.poses.filter((p) => p.status === "generating").length;
      if (!allowSharedJob && generatingCount > 0) continue;

      const poseIndex = claimablePoseIndex(job, now);
      if (poseIndex === -1) continue;

      const claimed = await jobs().findOneAndUpdate(
        {
          _id: job._id,
          status: { $in: [...ACTIVE_STATUSES] },
          [`poses.${poseIndex}.status`]: "pending",
        },
        {
          $set: {
            status: "running",
            startedAt: job.startedAt ?? now,
            [`poses.${poseIndex}.status`]: "generating",
            [`poses.${poseIndex}.startedAt`]: now,
          },
          $unset: { [`poses.${poseIndex}.error`]: "", [`poses.${poseIndex}.nextAttemptAt`]: "" },
          $inc: { [`poses.${poseIndex}.attempts`]: 1 },
        },
        { returnDocument: "after" },
      );

      if (claimed) return { job: claimed, poseIndex };
    }
  }

  return null;
};

const setPose = async (jobId: ObjectId, poseIndex: number, patch: Partial<JobPose>): Promise<void> => {
  const $set: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(patch)) $set[`poses.${poseIndex}.${k}`] = v;
  await jobs().updateOne({ _id: jobId }, { $set });
};

// Marks the job done/failed once no pose is pending or generating. A cancelled
// job stays cancelled; whatever poses finished are kept.
const finalizeJobIfComplete = async (jobId: ObjectId): Promise<void> => {
  const job = await jobs().findOne({ _id: jobId });
  if (!job) return;

  const unfinished = job.poses.some((p) => p.status === "pending" || p.status === "generating");
  if (unfinished) return;

  jobCache.delete(jobId.toHexString());

  if (job.status === "cancelled") return;

  const anyDone = job.poses.some((p) => p.status === "done");
  const failedCount = job.poses.filter((p) => p.status === "failed").length;
  await jobs().updateOne(
    { _id: jobId, status: { $in: [...ACTIVE_STATUSES] } },
    {
      $set: {
        status: anyDone ? "done" : "failed",
        finishedAt: new Date(),
        ...(failedCount ? { error: `${failedCount} pose(s) failed.` } : {}),
      },
    },
  );
  console.log(`  [job ${jobId}] ${anyDone ? "done" : "failed"} (${failedCount} failed pose(s))`);
};

const runPose = async ({ job, poseIndex }: Claim): Promise<void> => {
  const pose = job.poses[poseIndex];
  const tag = `[job ${job._id} / ${pose.title}]`;
  const startedAt = Date.now();
  console.log(`  ${tag} start (attempt ${pose.attempts}/${MAX_ATTEMPTS}, user ${job.userName} ${job.userLocation}${job.priority ? ", URGENT" : ""})`);

  try {
    // The job may have been cancelled between claim and now; don't spend a
    // Gemini call on it.
    const fresh = await jobs().findOne({ _id: job._id }, { projection: { status: 1, referenceKey: 1 } });
    if (!fresh || fresh.status === "cancelled") {
      await setPose(job._id, poseIndex, { status: "cancelled", finishedAt: new Date() });
      console.log(`  ${tag} skipped - job cancelled`);
      return;
    }

    const cache = cacheFor(job._id);
    cache.session ??= loadSession(job);
    const session = await cache.session;

    let reference: GeneratedImage | null = null;
    if (fresh.referenceKey) {
      const referencePose = job.poses.find((p) => p.key === fresh.referenceKey);
      cache.reference ??= loadReference(fresh.referenceKey, referencePose?.mimeType || "image/png");
      reference = await cache.reference;
    }

    const image = await generatePoseImage(getAi(), session, pose, reference, job.params.imageSize);

    // The job may have been cancelled and deleted while this pose rendered;
    // don't leave an orphan file in storage.
    if (!(await jobs().findOne({ _id: job._id }, { projection: { _id: 1 } }))) {
      console.log(`  ${tag} discarded - job was deleted while rendering`);
      return;
    }

    const key = objectKey(
      "jobs",
      job._id.toHexString(),
      "outputs",
      `${poseIndex + 1}-${slug(pose.title)}.${extensionForMimeType(image.mimeType)}`,
    );
    const original = Buffer.from(image.data, "base64");
    await putObject(key, original, image.mimeType);

    // Preview is best-effort: a thumbnail problem must never fail the pose.
    // The startup backfill picks up any pose left without one.
    let thumbKey: string | undefined;
    try {
      thumbKey = await createThumbnail(key, original);
    } catch (error) {
      console.warn(`  ${tag} preview failed - ${briefError(error)}`);
    }

    await setPose(job._id, poseIndex, {
      status: "done",
      key,
      ...(thumbKey ? { thumbKey } : {}),
      mimeType: image.mimeType,
      width: image.width,
      height: image.height,
      finishedAt: new Date(),
    });

    // First finished pose becomes the consistency reference for the rest.
    const becameReference = await jobs().updateOne(
      { _id: job._id, referenceKey: { $exists: false } },
      { $set: { referenceKey: key } },
    );
    if (becameReference.modifiedCount) {
      cache.reference = Promise.resolve(image);
    }

    const size = image.width && image.height ? ` ${image.width}x${image.height}` : "";
    console.log(`  ${tag} done in ${((Date.now() - startedAt) / 1000).toFixed(1)}s${size} (${job.params.imageSize} requested)`);
  } catch (error) {
    const message = extractErrorMessage(error);
    const retryable = isRetryableError(error) && pose.attempts < MAX_ATTEMPTS;

    if (retryable) {
      const wait = backoffMs(pose.attempts);
      await setPose(job._id, poseIndex, {
        status: "pending",
        error: message,
        nextAttemptAt: new Date(Date.now() + wait),
      });
      console.warn(`  ${tag} retryable failure after ${((Date.now() - startedAt) / 1000).toFixed(1)}s, retry in ${Math.round(wait / 1000)}s - ${message}`);
    } else {
      await setPose(job._id, poseIndex, { status: "failed", error: message, finishedAt: new Date() });
      console.error(`  ${tag} failed after ${((Date.now() - startedAt) / 1000).toFixed(1)}s (attempt ${pose.attempts}/${MAX_ATTEMPTS}) - ${briefError(error)}`);
    }
  } finally {
    await finalizeJobIfComplete(job._id).catch((error) => {
      console.error(`  ${tag} finalize error:`, error);
    });
  }
};

const tick = async (): Promise<void> => {
  if (scheduling) return;
  scheduling = true;

  try {
    while (activeSlots < CONCURRENCY) {
      const claim = await claimNextPose();
      if (!claim) break;

      activeSlots++;
      void runPose(claim)
        .catch((error) => console.error("  [worker] unexpected error:", error))
        .finally(() => {
          activeSlots--;
          // Re-check right away: the finished pose may have unblocked the
          // other poses of its job.
          setTimeout(() => void tick(), 1500);
        });
    }
  } catch (error) {
    console.error("  [worker] scheduling error:", error);
  } finally {
    scheduling = false;
  }
};

// Poses left "generating" by a crash or redeploy are put back in the queue.
const recoverInterruptedPoses = async (): Promise<void> => {
  const result = await jobs().updateMany(
    { status: { $in: [...ACTIVE_STATUSES] }, "poses.status": "generating" },
    { $set: { "poses.$[p].status": "pending" } },
    { arrayFilters: [{ "p.status": "generating" }] },
  );
  if (result.modifiedCount) {
    console.log(`  [worker] re-queued interrupted poses in ${result.modifiedCount} job(s)`);
  }
};

export const startWorker = async (): Promise<void> => {
  if (started) return;
  started = true;

  await recoverInterruptedPoses();
  console.log(`  [worker] started with ${CONCURRENCY} slot(s), max ${MAX_ATTEMPTS} attempt(s) per pose`);
  void backfillThumbnails();

  const loop = async () => {
    while (true) {
      await tick();
      await delay(POLL_MS);
    }
  };
  void loop();
};

// Called after a job is created so it doesn't wait for the next poll.
export const wakeWorker = (): void => {
  void tick();
};

export const workerStats = () => ({ concurrency: CONCURRENCY, activeSlots });
