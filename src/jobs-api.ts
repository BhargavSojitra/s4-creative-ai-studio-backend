import express from "express";
import {
  INPUT_IMAGE_FIELDS,
  isDbConfigured,
  isDbConnected,
  type JobDoc,
  type JobPose,
  jobs,
  ObjectId,
  type StoredImageRef,
  toObjectId,
  type UserDoc,
  users,
} from "./db";
import { AppError, type GenerateRequestBody, validateGenerateRequest } from "./gemini";
import { wakeWorker, workerStats } from "./queue";
import {
  assertStorageConfiguration,
  deletePrefix,
  extensionForMimeType,
  getObjectStream,
  objectKey,
  putObject,
  urlForKey,
} from "./storage";

type CreateJobBody = GenerateRequestBody & { userId?: string; urgent?: boolean };

type Handler = (req: express.Request, res: express.Response) => Promise<unknown>;

const ACTIVE_STATUSES = ["queued", "running"] as const;
const MAX_NAME_LENGTH = 60;

const asyncHandler = (fn: Handler): express.RequestHandler => (req, res, next) => {
  fn(req, res).catch(next);
};

const serializeUser = (user: UserDoc) => ({
  id: user._id.toHexString(),
  name: user.name,
  location: user.location,
  createdAt: user.createdAt,
});

const fileNameForPose = (pose: JobPose, index: number): string =>
  `${index + 1}-${pose.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "pose"}.${extensionForMimeType(pose.mimeType || "image/png")}`;

// `url` is the direct R2 link (display); `downloadPath` is served by this API
// with CORS headers so canvas/zip downloads work without bucket CORS rules.
const serializePose = async (pose: JobPose, index: number, jobId: ObjectId) => ({
  id: pose.id,
  index,
  title: pose.title,
  category: pose.category ?? null,
  status: pose.status,
  attempts: pose.attempts,
  width: pose.width ?? null,
  height: pose.height ?? null,
  error: pose.error ?? null,
  startedAt: pose.startedAt ?? null,
  finishedAt: pose.finishedAt ?? null,
  url: pose.status === "done" && pose.key ? await urlForKey(pose.key) : null,
  // Small WebP for lists/grids only; null until the preview exists.
  thumbUrl: pose.status === "done" && pose.thumbKey ? await urlForKey(pose.thumbKey) : null,
  downloadPath: pose.status === "done" && pose.key ? `/jobs/${jobId.toHexString()}/poses/${index}/file` : null,
  fileName: fileNameForPose(pose, index),
});

const serializeJob = async (job: JobDoc, position: number | null) => ({
  id: job._id.toHexString(),
  user: { id: job.userId.toHexString(), name: job.userName, location: job.userLocation },
  status: job.status,
  urgent: job.priority === 1,
  position,
  params: job.params,
  poses: await Promise.all(job.poses.map((pose, index) => serializePose(pose, index, job._id))),
  progress: {
    done: job.poses.filter((p) => p.status === "done").length,
    failed: job.poses.filter((p) => p.status === "failed").length,
    total: job.poses.length,
  },
  error: job.error ?? null,
  createdAt: job.createdAt,
  startedAt: job.startedAt ?? null,
  finishedAt: job.finishedAt ?? null,
});

// 1-based position of every active job in scheduling order (urgent first,
// then oldest). Running jobs are included so the UI can show them at the top.
const activePositions = async (): Promise<Map<string, number>> => {
  const active = await jobs()
    .find({ status: { $in: [...ACTIVE_STATUSES] } })
    .sort({ priority: -1, createdAt: 1 })
    .project<{ _id: ObjectId }>({ _id: 1 })
    .toArray();
  return new Map(active.map((job, index) => [job._id.toHexString(), index + 1]));
};

const serializeJobs = async (list: JobDoc[]) => {
  const positions = await activePositions();
  return Promise.all(list.map((job) => serializeJob(job, positions.get(job._id.toHexString()) ?? null)));
};

const requireJobId = (value: unknown): ObjectId => {
  const id = toObjectId(value);
  if (!id) throw new AppError("Invalid job id.", 400, "INVALID_JOB_ID");
  return id;
};

const findJobOrThrow = async (id: ObjectId): Promise<JobDoc> => {
  const job = await jobs().findOne({ _id: id });
  if (!job) throw new AppError("Job not found.", 404, "JOB_NOT_FOUND");
  return job;
};

const cleanName = (value: unknown, label: string): string => {
  const text = typeof value === "string" ? value.trim().replace(/\s+/g, " ") : "";
  if (!text) throw new AppError(`${label} is required.`, 400, "INVALID_USER");
  if (text.length > MAX_NAME_LENGTH) throw new AppError(`${label} must be at most ${MAX_NAME_LENGTH} characters.`, 400, "INVALID_USER");
  return text;
};

export const router = express.Router();

router.use((_req, _res, next) => {
  if (!isDbConfigured()) {
    return next(new AppError("Job queue is not configured on this server (MONGODB_URI missing).", 503, "QUEUE_DISABLED"));
  }
  if (!isDbConnected()) {
    return next(new AppError("Job queue database is not reachable right now, retrying in the background.", 503, "QUEUE_UNAVAILABLE"));
  }
  next();
});

router.get("/users", asyncHandler(async (_req, res) => {
  const list = await users().find().sort({ name: 1, location: 1 }).toArray();
  res.json(list.map(serializeUser));
}));

router.post("/users", asyncHandler(async (req, res) => {
  const name = cleanName(req.body?.name, "Name");
  const location = cleanName(req.body?.location, "Location");

  const existing = await users().findOne({ name, location }, { collation: { locale: "en", strength: 2 } });
  if (existing) return res.json(serializeUser(existing));

  const doc: UserDoc = { _id: new ObjectId(), name, location, createdAt: new Date() };
  await users().insertOne(doc);
  res.status(201).json(serializeUser(doc));
}));

router.post("/jobs", asyncHandler(async (req, res) => {
  const body = req.body as CreateJobBody;
  const urgent = Boolean(body.urgent);

  const userId = toObjectId(body.userId);
  if (!userId) throw new AppError("Select a user before generating.", 400, "MISSING_USER");
  const user = await users().findOne({ _id: userId });
  if (!user) throw new AppError("User not found.", 404, "USER_NOT_FOUND");

  validateGenerateRequest(body);
  assertStorageConfiguration();

  const jobId = new ObjectId();
  const inputs: Partial<Record<(typeof INPUT_IMAGE_FIELDS)[number], StoredImageRef>> = {};

  await Promise.all(
    INPUT_IMAGE_FIELDS.map(async (field) => {
      const image = body[field];
      if (!image?.data || !image?.mimeType) return;
      const key = objectKey("jobs", jobId.toHexString(), "inputs", `${field}.${extensionForMimeType(image.mimeType)}`);
      await putObject(key, Buffer.from(image.data, "base64"), image.mimeType);
      inputs[field] = { key, mimeType: image.mimeType };
    }),
  );

  const job: JobDoc = {
    _id: jobId,
    userId: user._id,
    userName: user.name,
    userLocation: user.location,
    status: "queued",
    priority: urgent ? 1 : 0,
    params: {
      modelAttributes: body.modelAttributes ?? "",
      backgroundPrompt: body.backgroundPrompt ?? "",
      imageSize: body.imageSize,
      aspectRatio: body.aspectRatio,
      expressionPrompt: body.expressionPrompt || undefined,
      necklinePrompt: body.necklinePrompt || undefined,
      sleevePrompt: body.sleevePrompt || undefined,
    },
    inputs,
    poses: body.selectedPoses.map((p) => ({
      id: p.id,
      title: p.title,
      prompt: p.prompt,
      category: p.category,
      status: "pending",
      attempts: 0,
    })),
    createdAt: new Date(),
  };

  await jobs().insertOne(job);
  wakeWorker();

  console.log(`  [job ${jobId}] queued by ${user.name} (${user.location}) - ${job.poses.length} pose(s), ${job.params.imageSize} ${job.params.aspectRatio}${job.priority ? ", URGENT" : ""}`);

  const [serialized] = await serializeJobs([job]);
  res.status(201).json(serialized);
}));

router.get("/queue", asyncHandler(async (_req, res) => {
  const active = await jobs()
    .find({ status: { $in: [...ACTIVE_STATUSES] } })
    .sort({ priority: -1, createdAt: 1 })
    .toArray();
  res.json({ worker: workerStats(), jobs: await serializeJobs(active) });
}));

router.get("/jobs", asyncHandler(async (req, res) => {
  const filter: Record<string, unknown> = {};

  if (typeof req.query.userId === "string") {
    const userId = toObjectId(req.query.userId);
    if (!userId) throw new AppError("Invalid userId.", 400, "INVALID_USER_ID");
    filter.userId = userId;
  }
  if (typeof req.query.status === "string") {
    filter.status = { $in: req.query.status.split(",").map((s) => s.trim()).filter(Boolean) };
  }

  const limit = Math.min(200, Math.max(1, Number(req.query.limit) || 50));
  const list = await jobs().find(filter).sort({ createdAt: -1 }).limit(limit).toArray();
  res.json(await serializeJobs(list));
}));

router.get("/jobs/:id", asyncHandler(async (req, res) => {
  const job = await findJobOrThrow(requireJobId(req.params.id));
  const [serialized] = await serializeJobs([job]);
  res.json(serialized);
}));

router.get("/jobs/:id/poses/:index/file", asyncHandler(async (req, res) => {
  const job = await findJobOrThrow(requireJobId(req.params.id));
  const index = Number(req.params.index);
  const pose = Number.isInteger(index) ? job.poses[index] : undefined;
  if (!pose || pose.status !== "done" || !pose.key) {
    throw new AppError("Image is not available for this pose.", 404, "IMAGE_NOT_READY");
  }

  const object = await getObjectStream(pose.key);
  res.setHeader("Content-Type", object.contentType || pose.mimeType || "image/png");
  if (object.contentLength) res.setHeader("Content-Length", String(object.contentLength));
  res.setHeader("Content-Disposition", `${req.query.download ? "attachment" : "inline"}; filename="${fileNameForPose(pose, index)}"`);
  res.setHeader("Cache-Control", "private, max-age=86400");
  object.stream.pipe(res);
}));

// Pending poses are dropped; a pose already generating is allowed to finish
// and its image is kept.
router.post("/jobs/:id/cancel", asyncHandler(async (req, res) => {
  const id = requireJobId(req.params.id);
  const result = await jobs().updateOne(
    { _id: id, status: { $in: [...ACTIVE_STATUSES] } },
    {
      $set: { status: "cancelled", finishedAt: new Date(), "poses.$[p].status": "cancelled" },
    },
    { arrayFilters: [{ "p.status": "pending" }] },
  );
  if (!result.matchedCount) {
    const job = await findJobOrThrow(id);
    throw new AppError(`Job is already ${job.status}.`, 409, "JOB_NOT_ACTIVE");
  }

  console.log(`  [job ${id}] cancelled by user`);
  const [serialized] = await serializeJobs([await findJobOrThrow(id)]);
  res.json(serialized);
}));

// Re-queues failed/cancelled poses of a finished job. Finished poses are kept
// and the existing reference image keeps the look consistent.
router.post("/jobs/:id/retry", asyncHandler(async (req, res) => {
  const id = requireJobId(req.params.id);
  const job = await findJobOrThrow(id);

  if ((ACTIVE_STATUSES as readonly string[]).includes(job.status)) {
    throw new AppError("Job is still active.", 409, "JOB_ACTIVE");
  }
  if (!job.poses.some((p) => p.status === "failed" || p.status === "cancelled")) {
    throw new AppError("Nothing to retry.", 409, "NOTHING_TO_RETRY");
  }

  await jobs().updateOne(
    { _id: id },
    {
      $set: {
        status: "queued",
        "poses.$[p].status": "pending",
        "poses.$[p].attempts": 0,
        ...(req.body?.urgent ? { priority: 1 } : {}),
      },
      $unset: {
        finishedAt: "",
        error: "",
        "poses.$[p].error": "",
        "poses.$[p].finishedAt": "",
        "poses.$[p].nextAttemptAt": "",
      },
    },
    { arrayFilters: [{ "p.status": { $in: ["failed", "cancelled"] } }] },
  );
  wakeWorker();
  console.log(`  [job ${id}] retry requested - ${job.poses.filter((p) => p.status === "failed" || p.status === "cancelled").length} pose(s) re-queued${req.body?.urgent ? ", URGENT" : ""}`);

  const [serialized] = await serializeJobs([await findJobOrThrow(id)]);
  res.json(serialized);
}));

// Permanently removes a finished job: its inputs and outputs in R2 and the
// job document. Used once the user has downloaded what they need, so storage
// does not grow forever. Active jobs must be cancelled first.
router.delete("/jobs/:id", asyncHandler(async (req, res) => {
  const id = requireJobId(req.params.id);
  const job = await findJobOrThrow(id);

  if ((ACTIVE_STATUSES as readonly string[]).includes(job.status)) {
    throw new AppError("Cancel the job before deleting it.", 409, "JOB_ACTIVE");
  }

  const removed = await deletePrefix(`${objectKey("jobs", id.toHexString())}/`);
  await jobs().deleteOne({ _id: id });
  console.log(`  [job ${id}] deleted by user (${removed} file(s) removed from storage)`);

  res.json({ id: id.toHexString(), deleted: true, filesRemoved: removed });
}));
