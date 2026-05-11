import cors from "cors";
import express from "express";
import {
  AppError,
  assertServerConfiguration,
  extractErrorMessage,
  streamPhotoshootImages,
  type GenerateRequestBody,
  validateGenerateRequest,
} from "./gemini";

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
const LOG_DIVIDER = "\n-----------------------------------------";
const LOG_DIVIDER_END = "-----------------------------------------\n";

type RequestWithId = express.Request & {
  requestId?: string;
  requestStartTime?: number;
};

const summarizeBodyForLogs = (body: GenerateRequestBody) => Object.fromEntries(
  Object.entries(body).map(([k, v]) => {
    if (k.toLowerCase().includes("image") && v && typeof v === "object" && "data" in (v as object)) {
      return [k, { mimeType: (v as { mimeType: string }).mimeType, dataLength: (v as { data: string }).data?.length }];
    }
    return [k, v];
  }),
);

app.use((req: RequestWithId, res, next) => {
  const start = Date.now();
  const requestId = `${start.toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

  req.requestId = requestId;
  req.requestStartTime = start;

  console.log(LOG_DIVIDER);
  console.log(`[${new Date(start).toISOString()}] ${req.method} ${req.originalUrl} [${requestId}] start`);

  res.on("finish", () => {
    const durationSeconds = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} [${requestId}] end status=${res.statusCode} duration=${durationSeconds}s`);
    console.log(LOG_DIVIDER_END);
  });

  res.on("close", () => {
    if (!res.writableEnded) {
      const durationSeconds = ((Date.now() - start) / 1000).toFixed(1);
      console.error(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} [${requestId}] closed early status=${res.statusCode} duration=${durationSeconds}s`);
      console.log(LOG_DIVIDER_END);
    }
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
    methods: ["POST", "OPTIONS"],
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

  console.log(`  [${requestId}] poses     : ${body.selectedPoses?.map((p) => p.title).join(", ") || "none"}`);
  console.log(`  [${requestId}] imageSize : ${body.imageSize}  aspectRatio: ${body.aspectRatio}`);
  console.log(`  [${requestId}] garment   : ${body.garmentImage ? "yes" : "no"}  model: ${body.modelImage ? "yes" : "no"}  bg: ${body.backgroundImage ? "yes" : "no"}`);
  console.log(`  [${requestId}] extras    : design=${body.designDetailImage ? "yes" : "no"} styling=${body.stylingImage ? "yes" : "no"} dupatta=${body.dupattaImage ? "yes" : "no"} blouse=${body.blouseImage ? "yes" : "no"} back=${body.backViewImage ? "yes" : "no"}`);
  console.log(`  [${requestId}] [DEBUG] body keys:`, Object.keys(body));
  console.log(`  [${requestId}] [DEBUG] selectedPoses raw:`, JSON.stringify(body.selectedPoses));
  console.log(`  [${requestId}] [DEBUG] full body (images stripped):`, JSON.stringify(summarizeBodyForLogs(body), null, 2));

  try {
    validateGenerateRequest(body);
    assertServerConfiguration();
  } catch (error) {
    const msg = extractErrorMessage(error);
    console.error(`  [${requestId}] error before streaming after ${((Date.now() - start) / 1000).toFixed(1)}s - ${msg}`);
    console.error(`  [${requestId}] raw:`, error);
    return sendJsonError(res, error);
  }

  res.setHeader("Content-Type", "application/x-ndjson");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");

  let count = 0;
  try {
    for await (const image of streamPhotoshootImages(body)) {
      res.write(JSON.stringify(image) + "\n");
      count++;
      console.log(`  [${requestId}] streamed [${image.poseTitle}] (${count}/${body.selectedPoses.length}) - ${((Date.now() - start) / 1000).toFixed(1)}s`);
    }

    console.log(`  [${requestId}] done in ${((Date.now() - start) / 1000).toFixed(1)}s - ${count} image(s) streamed`);
  } catch (error) {
    const msg = extractErrorMessage(error);
    console.error(`  [${requestId}] error after ${((Date.now() - start) / 1000).toFixed(1)}s - ${msg}`);
    console.error(`  [${requestId}] raw:`, error);
    res.write(JSON.stringify({ error: msg }) + "\n");
  }

  res.end();
});

app.use((error: unknown, req: RequestWithId, res: express.Response, _next: express.NextFunction) => {
  const requestId = req.requestId ?? "unknown";
  console.error(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} [${requestId}] unhandled error: ${extractErrorMessage(error)}`);
  console.error(`  [${requestId}] raw:`, error);

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

app.listen(PORT, () => {
  console.log(`S4 Textile Backend running on port ${PORT}`);
  console.log(`Host: ${HOST}`);
  console.log(`Base URL: ${BASE_URL}`);
});
