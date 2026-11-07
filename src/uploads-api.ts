import express from "express";
import multer from "multer";
import { ObjectId } from "mongodb";
import { AppError } from "./gemini";
import {
  assertStorageConfiguration,
  deleteStaleObjects,
  extensionForMimeType,
  isStorageConfigured,
  objectKey,
  putObject,
} from "./storage";

// Reference images reach the server as multipart files, one per request, and
// are staged under `uploads/` in R2. POST /jobs then refers to them by key and
// copies them into the job folder, so the JSON job body never carries image
// bytes (base64 bodies used to exceed the body limit and take minutes).

const MAX_FILE_MB = Number(process.env.UPLOAD_MAX_FILE_MB) || 60;
const MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024;
const ALLOWED_MIME_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
// Staged files a job never claimed are removed after this long.
const STALE_UPLOAD_MS = 24 * 60 * 60 * 1000;
const SWEEP_INTERVAL_MS = 6 * 60 * 60 * 1000;

export const UPLOADS_PREFIX = objectKey("uploads");

export type UploadedImageRef = { key: string; mimeType: string };

export const isUploadedImageRef = (value: unknown): value is UploadedImageRef =>
  typeof value === "object"
  && value !== null
  && typeof (value as UploadedImageRef).key === "string"
  && typeof (value as UploadedImageRef).mimeType === "string";

/** Only keys this endpoint produced may be referenced by a job (no arbitrary bucket reads). */
export const isStagedUploadKey = (key: string): boolean =>
  key.startsWith(`${UPLOADS_PREFIX}/`) && !key.includes("..") && /^[A-Za-z0-9_\-./]+$/.test(key);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_BYTES, files: 1, fields: 0 },
  fileFilter: (_req, file, cb) => {
    if (!ALLOWED_MIME_TYPES.has(file.mimetype)) {
      return cb(new AppError(`Unsupported image type ${file.mimetype}. Use PNG, JPEG or WebP.`, 415, "UNSUPPORTED_IMAGE_TYPE"));
    }
    cb(null, true);
  },
});

const single = upload.single("file");

export const router = express.Router();

router.post("/uploads", (req, res, next) => {
  if (!isStorageConfigured()) {
    return next(new AppError("Uploads are not configured on this server (R2 missing).", 503, "STORAGE_DISABLED"));
  }

  single(req, res, (error: unknown) => {
    if (error instanceof multer.MulterError) {
      if (error.code === "LIMIT_FILE_SIZE") {
        return next(new AppError(`Image is larger than ${MAX_FILE_MB} MB.`, 413, "UPLOAD_TOO_LARGE"));
      }
      return next(new AppError(`Invalid upload: ${error.message}.`, 400, "INVALID_UPLOAD"));
    }
    if (error) return next(error);

    const file = req.file;
    if (!file || !file.buffer?.length) {
      return next(new AppError('Attach the image as multipart field "file".', 400, "MISSING_FILE"));
    }

    const key = objectKey("uploads", `${new ObjectId().toHexString()}.${extensionForMimeType(file.mimetype)}`);
    putObject(key, file.buffer, file.mimetype)
      .then(() => res.status(201).json({ key, mimeType: file.mimetype, size: file.size }))
      .catch(next);
  });
});

let sweepRunning = false;

export const sweepStaleUploads = async (): Promise<void> => {
  if (sweepRunning) return;
  sweepRunning = true;
  try {
    assertStorageConfiguration();
    const removed = await deleteStaleObjects(`${UPLOADS_PREFIX}/`, new Date(Date.now() - STALE_UPLOAD_MS));
    if (removed) console.log(`  [uploads] removed ${removed} stale staged upload(s)`);
  } catch (error) {
    console.warn(`  [uploads] sweep failed - ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    sweepRunning = false;
  }
};

/** Runs the stale-upload sweep now and every few hours. */
export const scheduleUploadSweep = (): void => {
  void sweepStaleUploads();
  setInterval(() => void sweepStaleUploads(), SWEEP_INTERVAL_MS).unref();
};
