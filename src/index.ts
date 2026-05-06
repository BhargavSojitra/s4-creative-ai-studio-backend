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
const LOG_DIVIDER = "\n-----------------------------------------";
const LOG_DIVIDER_END = "-----------------------------------------\n";

// Allow large base64 payloads; no Vercel 4.5MB limit here.
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

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

app.post("/generate-photoshoot", async (req, res) => {
  const start = Date.now();
  const body = req.body as GenerateRequestBody;

  console.log(LOG_DIVIDER);
  console.log(`[${new Date().toISOString()}] POST /generate-photoshoot`);
  console.log(`  poses     : ${body.selectedPoses?.map((p) => p.title).join(", ") || "none"}`);
  console.log(`  imageSize : ${body.imageSize}  aspectRatio: ${body.aspectRatio}`);
  console.log(`  garment   : ${body.garmentImage ? "yes" : "no"}  model: ${body.modelImage ? "yes" : "no"}  bg: ${body.backgroundImage ? "yes" : "no"}`);
  console.log(`  extras    : design=${body.designDetailImage ? "yes" : "no"} styling=${body.stylingImage ? "yes" : "no"} dupatta=${body.dupattaImage ? "yes" : "no"} blouse=${body.blouseImage ? "yes" : "no"} back=${body.backViewImage ? "yes" : "no"}`);

  try {
    validateGenerateRequest(body);
    assertServerConfiguration();
  } catch (error) {
    const msg = extractErrorMessage(error);
    console.error(`  error before streaming after ${((Date.now() - start) / 1000).toFixed(1)}s - ${msg}`);
    console.error("  raw:", error);
    console.log(LOG_DIVIDER_END);
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
      console.log(`  streamed [${image.poseTitle}] (${count}/${body.selectedPoses.length}) - ${((Date.now() - start) / 1000).toFixed(1)}s`);
    }

    console.log(`  done in ${((Date.now() - start) / 1000).toFixed(1)}s - ${count} image(s) streamed`);
  } catch (error) {
    const msg = extractErrorMessage(error);
    console.error(`  error after ${((Date.now() - start) / 1000).toFixed(1)}s - ${msg}`);
    console.error("  raw:", error);
    res.write(JSON.stringify({ error: msg }) + "\n");
  }

  res.end();
  console.log(LOG_DIVIDER_END);
});

app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  if (res.headersSent) return;

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
