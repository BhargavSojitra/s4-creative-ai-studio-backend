import { type Collection, type Db, MongoClient, ObjectId } from "mongodb";
import type { GenerateRequestBody } from "./gemini";

export type UserDoc = {
  _id: ObjectId;
  name: string;
  location: string;
  createdAt: Date;
};

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type PoseStatus = "pending" | "generating" | "done" | "failed" | "cancelled";

export type StoredImageRef = { key: string; mimeType: string };

export type InputImageField =
  | "garmentImage"
  | "modelImage"
  | "backgroundImage"
  | "designDetailImage"
  | "stylingImage"
  | "dupattaImage"
  | "blouseImage"
  | "backViewImage"
  | "unstitchedSleeveImage"
  | "unstitchedCloseupImage"
  | "dummyShootImage";

export const INPUT_IMAGE_FIELDS: InputImageField[] = [
  "garmentImage",
  "modelImage",
  "backgroundImage",
  "designDetailImage",
  "stylingImage",
  "dupattaImage",
  "blouseImage",
  "backViewImage",
  "unstitchedSleeveImage",
  "unstitchedCloseupImage",
  "dummyShootImage",
];

export type JobParams = Pick<
  GenerateRequestBody,
  "modelAttributes" | "backgroundPrompt" | "imageSize" | "aspectRatio" | "expressionPrompt" | "necklinePrompt" | "sleevePrompt"
>;

export type JobPose = {
  id: string;
  title: string;
  prompt: string;
  category?: string;
  status: PoseStatus;
  attempts: number;
  key?: string;
  // Small WebP preview of `key` for the UI; downloads always use `key`.
  thumbKey?: string;
  mimeType?: string;
  width?: number;
  height?: number;
  error?: string;
  startedAt?: Date;
  finishedAt?: Date;
  nextAttemptAt?: Date;
};

export type JobDoc = {
  _id: ObjectId;
  userId: ObjectId;
  userName: string;
  userLocation: string;
  status: JobStatus;
  // 1 = urgent: runs before every normal queued job, never interrupts a running pose.
  priority: 0 | 1;
  params: JobParams;
  inputs: Partial<Record<InputImageField, StoredImageRef>>;
  poses: JobPose[];
  // R2 key of the first finished pose; every later pose is conditioned on it.
  referenceKey?: string;
  error?: string;
  createdAt: Date;
  startedAt?: Date;
  finishedAt?: Date;
};

let client: MongoClient | null = null;
let db: Db | null = null;

export const isDbConfigured = (): boolean => Boolean(process.env.MONGODB_URI);

export const isDbConnected = (): boolean => db !== null;

export const connectDb = async (): Promise<Db> => {
  if (db) return db;

  const uri = process.env.MONGODB_URI;
  if (!uri) throw new Error("MONGODB_URI is not set.");

  const candidate = new MongoClient(uri, { serverSelectionTimeoutMS: 15000 });
  try {
    await candidate.connect();
  } catch (error) {
    await candidate.close().catch(() => undefined);
    throw error;
  }
  client = candidate;
  db = client.db(process.env.MONGODB_DB || undefined);

  await Promise.all([
    users().createIndex({ name: 1, location: 1 }, { unique: true, collation: { locale: "en", strength: 2 } }),
    jobs().createIndex({ status: 1, priority: -1, createdAt: 1 }),
    jobs().createIndex({ userId: 1, createdAt: -1 }),
    jobs().createIndex({ createdAt: -1 }),
  ]);

  console.log(`  [db] connected to MongoDB database "${db.databaseName}"`);
  return db;
};

const requireDb = (): Db => {
  if (!db) throw new Error("Database not connected. Call connectDb() first.");
  return db;
};

export const users = (): Collection<UserDoc> => requireDb().collection<UserDoc>("users");
export const jobs = (): Collection<JobDoc> => requireDb().collection<JobDoc>("jobs");

export const toObjectId = (value: unknown): ObjectId | null => {
  if (typeof value !== "string" || !ObjectId.isValid(value)) return null;
  return new ObjectId(value);
};

export { ObjectId };
