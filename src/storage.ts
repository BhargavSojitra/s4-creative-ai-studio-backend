import {
  DeleteObjectsCommand,
  GetObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const R2_ACCOUNT_ID = process.env.R2_ACCOUNT_ID;
const R2_BUCKET = process.env.R2_BUCKET;
const R2_ACCESS_KEY_ID = process.env.R2_ACCESS_KEY_ID;
const R2_SECRET_ACCESS_KEY = process.env.R2_SECRET_ACCESS_KEY;
// Optional public bucket URL (r2.dev or custom domain). Without it, download
// links are presigned and expire.
const R2_PUBLIC_BASE_URL = (process.env.R2_PUBLIC_BASE_URL || "").replace(/\/+$/, "");
// Lets several apps share one bucket without colliding.
const R2_KEY_PREFIX = (process.env.R2_KEY_PREFIX || "").replace(/^\/+|\/+$/g, "");
const SIGNED_URL_TTL_SECONDS = 60 * 60;

let s3: S3Client | null = null;

export const isStorageConfigured = (): boolean =>
  Boolean(R2_ACCOUNT_ID && R2_BUCKET && R2_ACCESS_KEY_ID && R2_SECRET_ACCESS_KEY);

export const assertStorageConfiguration = (): void => {
  if (!isStorageConfigured()) {
    throw new Error(
      "Missing R2 configuration. Set R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.",
    );
  }
};

const getClient = (): S3Client => {
  if (s3) return s3;
  assertStorageConfiguration();
  s3 = new S3Client({
    region: "auto",
    endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    credentials: { accessKeyId: R2_ACCESS_KEY_ID!, secretAccessKey: R2_SECRET_ACCESS_KEY! },
  });
  return s3;
};

export const objectKey = (...parts: string[]): string =>
  [R2_KEY_PREFIX, ...parts].filter(Boolean).join("/");

export const extensionForMimeType = (mimeType: string): string => {
  const subtype = mimeType.split("/")[1]?.toLowerCase() || "bin";
  return subtype === "jpeg" ? "jpg" : subtype.replace(/[^a-z0-9]/g, "") || "bin";
};

export const putObject = async (key: string, body: Buffer, contentType: string): Promise<void> => {
  await getClient().send(new PutObjectCommand({ Bucket: R2_BUCKET, Key: key, Body: body, ContentType: contentType }));
};

/** Deletes every object under `prefix` (inputs and outputs of one job). Returns the count. */
export const deletePrefix = async (prefix: string): Promise<number> => {
  const client = getClient();
  let deleted = 0;
  let token: string | undefined;
  do {
    const page = await client.send(new ListObjectsV2Command({ Bucket: R2_BUCKET, Prefix: prefix, ContinuationToken: token }));
    const keys = (page.Contents ?? []).flatMap((o) => (o.Key ? [{ Key: o.Key }] : []));
    if (keys.length) {
      await client.send(new DeleteObjectsCommand({ Bucket: R2_BUCKET, Delete: { Objects: keys, Quiet: true } }));
      deleted += keys.length;
    }
    token = page.IsTruncated ? page.NextContinuationToken : undefined;
  } while (token);
  return deleted;
};

export const getObject = async (key: string): Promise<Buffer> => {
  const result = await getClient().send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }));
  if (!result.Body) throw new Error(`Empty object body for ${key}`);
  return Buffer.from(await result.Body.transformToByteArray());
};

export type StoredObjectStream = {
  stream: NodeJS.ReadableStream;
  contentType?: string;
  contentLength?: number;
};

// Streams an object without buffering it, for the download proxy route.
export const getObjectStream = async (key: string): Promise<StoredObjectStream> => {
  const result = await getClient().send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }));
  if (!result.Body) throw new Error(`Empty object body for ${key}`);
  return {
    stream: result.Body as unknown as NodeJS.ReadableStream,
    contentType: result.ContentType,
    contentLength: result.ContentLength,
  };
};

export const urlForKey = async (key: string): Promise<string> => {
  if (R2_PUBLIC_BASE_URL) return `${R2_PUBLIC_BASE_URL}/${key}`;
  return getSignedUrl(getClient(), new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }), {
    expiresIn: SIGNED_URL_TTL_SECONDS,
  });
};
