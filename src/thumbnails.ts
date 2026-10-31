import sharp from "sharp";
import { type JobPose, jobs } from "./db";
import { getObject, putObject } from "./storage";

// Small WebP previews for the UI. The original PNG is untouched and stays the
// only thing users download, view full-size, edit or zip. Without previews the
// job list pulls the full 4K PNG (~25 MB) for every 40px thumbnail.
const THUMB_MAX_SIDE = 640;
const THUMB_QUALITY = 82;
const THUMB_MIME = "image/webp";

// jobs/<id>/outputs/3-pose.png -> jobs/<id>/outputs/thumbs/3-pose.webp
// Derived from the stored key (not objectKey) so the key prefix the job was
// created under is preserved, and DELETE /jobs/:id removes it with the job.
export const thumbKeyFor = (key: string): string => {
  const slash = key.lastIndexOf("/");
  const dir = key.slice(0, slash + 1);
  const file = key.slice(slash + 1).replace(/\.[^.]+$/, "");
  return `${dir}thumbs/${file}.webp`;
};

export const makeThumbnail = (original: Buffer): Promise<Buffer> =>
  sharp(original)
    .rotate()
    .resize(THUMB_MAX_SIDE, THUMB_MAX_SIDE, { fit: "inside", withoutEnlargement: true })
    .webp({ quality: THUMB_QUALITY })
    .toBuffer();

/** Writes the preview for an original already in storage; returns its key. */
export const createThumbnail = async (key: string, original?: Buffer): Promise<string> => {
  const source = original ?? (await getObject(key));
  const thumbKey = thumbKeyFor(key);
  await putObject(thumbKey, await makeThumbnail(source), THUMB_MIME);
  return thumbKey;
};

let backfillRunning = false;

/**
 * Adds previews to poses finished before this feature existed. Runs once in
 * the background after startup, one image at a time so it never competes with
 * generation for memory. Safe to re-run: only poses without a thumbKey count.
 */
export const backfillThumbnails = async (): Promise<void> => {
  if (backfillRunning) return;
  backfillRunning = true;
  try {
    const pending = await jobs()
      .find({ poses: { $elemMatch: { status: "done", key: { $exists: true }, thumbKey: { $exists: false } } } })
      .project<{ _id: import("mongodb").ObjectId; poses: JobPose[] }>({ _id: 1, poses: 1 })
      .toArray();

    let count = 0;
    let failed = 0;
    for (const job of pending) {
      for (const [index, pose] of job.poses.entries()) {
        if (pose.status !== "done" || !pose.key || pose.thumbKey) continue;
        try {
          const thumbKey = await createThumbnail(pose.key);
          await jobs().updateOne({ _id: job._id }, { $set: { [`poses.${index}.thumbKey`]: thumbKey } });
          count++;
        } catch (error) {
          failed++;
          console.warn(`  [thumbs] backfill failed for ${pose.key} - ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
    if (count || failed) console.log(`  [thumbs] backfill done - ${count} preview(s) created${failed ? `, ${failed} failed` : ""}`);
  } catch (error) {
    console.warn(`  [thumbs] backfill aborted - ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    backfillRunning = false;
  }
};
