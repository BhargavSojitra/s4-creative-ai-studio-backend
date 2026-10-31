import cors from "cors";
import express from "express";
import { connectDb, isDbConfigured } from "./db";
import {
  AppError,
  assertServerConfiguration,
  briefError,
  extractErrorMessage,
  streamPhotoshootImages,
  type GenerateRequestBody,
  validateGenerateRequest,
} from "./gemini";
import { router as jobsRouter } from "./jobs-api";
import { startWorker } from "./queue";
import { isStorageConfigured } from "./storage";

const app = express();
const PORT = process.env.PORT || 8080;
const HOST = process.env.HOST || "0.0.0.0";
const RAILWAY_PUBLIC_DOMAIN = process.env.RAILWAY_PUBLIC_DOMAIN;
const RAILWAY_STATIC_URL = process.env.RAILWAY_STATIC_URL;
const BASE_URL = process.env.BASE_URL
  || (RAILWAY_STATIC_URL
    ? RAILWAY_STATIC_URL
    : RAILWAY_PUBLIC_DOMAIN
      ? `https://${RAILWAY_PUBLIC_DOMAIN}`
      : `http://localhost:${PORT}`);
const REQUEST_BODY_LIMIT = process.env.REQUEST_BODY_LIMIT || "100mb";
// Set LOG_REQUESTS=all to log every request (including the frontend's 3s polling).
const LOG_ALL_REQUESTS = process.env.LOG_REQUESTS === "all";

type RequestWithId = express.Request & {
  requestId?: string;
  requestStartTime?: number;
};

// Quiet request log: polling GETs, OPTIONS and health checks are skipped unless
// they fail. Writes (POST) and any 4xx/5xx are logged in one line each.
app.use((req: RequestWithId, res, next) => {
  const start = Date.now();
  const requestId = `${start.toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

  req.requestId = requestId;
  req.requestStartTime = start;

  const isWrite = req.method !== "GET" && req.method !== "OPTIONS" && req.method !== "HEAD";
  let logged = false;
  const logLine = (note: string, level: "log" | "error") => {
    if (logged) return;
    logged = true;
    const durationSeconds = ((Date.now() - start) / 1000).toFixed(1);
    console[level](`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} ${note} status=${res.statusCode} ${durationSeconds}s [${requestId}]`);
  };

  res.on("finish", () => {
    if (res.statusCode >= 400) logLine("ERROR", "error");
    else if (LOG_ALL_REQUESTS) logLine("ok", "log");
  });

  res.on("close", () => {
    // A generation stream cut off by the client is worth knowing about; polling GETs are not.
    if (!res.writableEnded && (isWrite || LOG_ALL_REQUESTS)) logLine("closed-early", "error");
  });

  next();
});

// Allow large base64 payloads; keep limit configurable for different deploy targets.
app.use(express.json({ limit: REQUEST_BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: REQUEST_BODY_LIMIT }));

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "http://localhost:3000")
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      callback(new AppError(`CORS origin ${origin} is not allowed.`, 403, "FORBIDDEN_ORIGIN"));
    },
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  }),
);

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.get("/helth", (_req, res) => {
  res.json({ status: "running" });
});

const sendJsonError = (res: express.Response, error: unknown) => {
  const statusCode = error instanceof AppError ? error.statusCode : 500;
  return res.status(statusCode).json({ error: extractErrorMessage(error) });
};

app.post("/generate-photoshoot", async (req: RequestWithId, res) => {
  const start = req.requestStartTime ?? Date.now();
  const requestId = req.requestId ?? "unknown";
  const body = req.body as GenerateRequestBody;

  const imageKeys = Object.entries(body)
    .filter(([k, v]) => k.toLowerCase().includes("image") && v && typeof v === "object")
    .map(([k]) => k.replace(/Image$/, ""));
  console.log(`  [${requestId}] legacy generate: ${body.selectedPoses?.length ?? 0} pose(s) [${body.selectedPoses?.map((p) => p.title).join(", ") || "none"}] ${body.imageSize} ${body.aspectRatio}, images: ${imageKeys.join(", ") || "none"}`);

  try {
    validateGenerateRequest(body);
    assertServerConfiguration();
  } catch (error) {
    const msg = extractErrorMessage(error);
    console.error(`  [${requestId}] rejected before streaming - ${msg}`);
    return sendJsonError(res, error);
  }

  res.setHeader("Content-Type", "application/x-ndjson");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");

  let clientConnected = true;
  res.on("close", () => {
    clientConnected = false;
  });

  let count = 0;
  try {
    for await (const image of streamPhotoshootImages(body, () => clientConnected)) {
      res.write(JSON.stringify(image) + "\n");
      count++;
      console.log(`  [${requestId}] streamed [${image.poseTitle}] (${count}/${body.selectedPoses.length}) - ${((Date.now() - start) / 1000).toFixed(1)}s`);
    }

    console.log(`  [${requestId}] done in ${((Date.now() - start) / 1000).toFixed(1)}s - ${count} image(s) streamed`);
  } catch (error) {
    const msg = extractErrorMessage(error);
    console.error(`  [${requestId}] failed after ${((Date.now() - start) / 1000).toFixed(1)}s (${count}/${body.selectedPoses.length} streamed) - ${briefError(error)}`);
    res.write(JSON.stringify({ error: msg }) + "\n");
  }

  res.end();
});

// Queue-based generation: POST /jobs, GET /queue, GET /jobs, GET /jobs/:id,
// POST /jobs/:id/cancel, POST /jobs/:id/retry, GET|POST /users.
app.use(jobsRouter);

app.use((error: unknown, req: RequestWithId, res: express.Response, _next: express.NextFunction) => {
  const requestId = req.requestId ?? "unknown";
  // Expected client errors (validation, not found, conflicts) are summarised by
  // the request logger's one-line ERROR entry; only real failures get a stack.
  const isClientError = error instanceof AppError && error.statusCode < 500;
  if (!isClientError) {
    console.error(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} [${requestId}] unhandled error: ${briefError(error)}`);
  }

  if (res.headersSent) return;

  if (
    typeof error === "object"
    && error !== null
    && "type" in error
    && error.type === "entity.too.large"
  ) {
    return res.status(413).json({
      error: `PAYLOAD_TOO_LARGE: Request body exceeds the configured limit of ${REQUEST_BODY_LIMIT}.`,
    });
  }

  if (error instanceof SyntaxError && "body" in error) {
    return res.status(400).json({ error: "INVALID_JSON: Request body contains invalid JSON." });
  }

  return sendJsonError(res, error);
});

const bootstrap = async () => {
  if (isDbConfigured()) {
    await connectDb();
    if (isStorageConfigured()) {
      await startWorker();
    } else {
      console.warn("  [worker] R2 is not configured - jobs can be queued but will not be processed.");
    }
  } else {
    console.warn("  [worker] MONGODB_URI not set - job queue disabled, only /generate-photoshoot is available.");
  }

  app.listen(PORT, () => {
    console.log(`S4 Textile Backend running on port ${PORT}`);
    console.log(`Host: ${HOST}`);
    console.log(`Base URL: ${BASE_URL}`);
  });
};

bootstrap().catch((error) => {
  console.error("Failed to start:", extractErrorMessage(error));
  console.error(error);
  process.exit(1);
});
