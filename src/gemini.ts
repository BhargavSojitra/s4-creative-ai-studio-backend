import { GoogleGenAI } from "@google/genai";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

const MODEL_NAME =
  process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image-preview";

type Pose = { id: string; title: string; prompt: string };

type InlineImagePayload = { data: string; mimeType: string };

export type GenerateRequestBody = {
  garmentImage: InlineImagePayload | null;
  modelAttributes: string;
  selectedPoses: Pose[];
  backgroundPrompt: string;
  imageSize: "1K" | "2K" | "4K";
  aspectRatio: "3:4" | "1:1" | "16:9" | "9:16";
  modelImage?: InlineImagePayload | null;
  backgroundImage?: InlineImagePayload | null;
  designDetailImage?: InlineImagePayload | null;
  stylingImage?: InlineImagePayload | null;
  dupattaImage?: InlineImagePayload | null;
  blouseImage?: InlineImagePayload | null;
  backViewImage?: InlineImagePayload | null;
};

export type GeneratedImagePayload = { poseTitle: string; url: string };
export type StreamChunk = GeneratedImagePayload | { error: string };
export class AppError extends Error {
  statusCode: number;
  code: string;
  expose: boolean;

  constructor(message: string, statusCode = 500, code = "INTERNAL_ERROR", expose = true) {
    super(message);
    this.name = "AppError";
    this.statusCode = statusCode;
    this.code = code;
    this.expose = expose;
  }
}

const delay = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

const imagePart = (image: InlineImagePayload) => ({
  inlineData: { data: image.data, mimeType: image.mimeType },
});

export const extractErrorMessage = (error: unknown): string => {
  if (error instanceof AppError) return `${error.code}: ${error.message}`;
  if (!error) return "An unexpected error occurred.";
  if (typeof error === "string") return error;

  const err = error as Record<string, unknown>;
  const status = (err.status || (err.error as Record<string, unknown>)?.code || err.code) as number | undefined;
  let message = String(err.message || (err.error as Record<string, unknown>)?.message || "");

  if (message.trim().startsWith("{")) {
    try {
      const parsed = JSON.parse(message);
      message = parsed.error?.message || parsed.message || message;
    } catch {}
  }

  const lowMsg = message.toLowerCase();

  if (status === 429 || lowMsg.includes("rate limit") || lowMsg.includes("quota"))
    return "QUOTA_EXCEEDED: You've reached the API rate limit. Please wait 1-2 minutes or try generating fewer poses at once.";

  if (status === 403 || lowMsg.includes("permission") || lowMsg.includes("api_key_invalid"))
    return "PERMISSION_DENIED: Check that billing is enabled, Vertex AI API is enabled, and the service account has the Vertex AI User role.";

  if (status === 404 || lowMsg.includes("requested entity was not found"))
    return `NOT_FOUND: The model "${MODEL_NAME}" was not found. Check GEMINI_IMAGE_MODEL and GOOGLE_CLOUD_LOCATION.`;

  if ((status as number) >= 500 || lowMsg.includes("overloaded") || lowMsg.includes("internal error") || lowMsg.includes("unavailable"))
    return "SERVER_ERROR: The AI servers are currently overloaded. Please try again.";

  if (lowMsg.includes("safety"))
    return "SAFETY_BLOCK: The request was flagged by safety filters. Try a different garment image.";

  return message || `Error (${status || "Unknown"}): Please check your Vertex AI setup.`;
};

const ensureServiceAccountFile = (): void => {
  const base64 = process.env.GOOGLE_SERVICE_ACCOUNT_BASE64;
  if (!base64 || process.env.GOOGLE_APPLICATION_CREDENTIALS) return;

  const keyDir = join(tmpdir(), "s4-textile-backend");
  const keyPath = join(keyDir, "gcp-key.json");

  try {
    const json = Buffer.from(base64, "base64").toString("utf8");
    JSON.parse(json);
    if (!existsSync(keyDir)) mkdirSync(keyDir, { recursive: true });
    writeFileSync(keyPath, json, "utf8");
    process.env.GOOGLE_APPLICATION_CREDENTIALS = keyPath;
    console.log("  [client] Service account written to", keyPath);
  } catch (error) {
    throw new AppError(
      `Unable to prepare Google credentials: ${extractErrorMessage(error)}`,
      500,
      "CREDENTIAL_SETUP_FAILED",
    );
  }
};

export function validateGenerateRequest(
  body: Partial<GenerateRequestBody> | null | undefined,
): asserts body is GenerateRequestBody & { garmentImage: InlineImagePayload; selectedPoses: Pose[] } {
  if (!body || typeof body !== "object") {
    throw new AppError("Request body must be a JSON object.", 400, "INVALID_BODY");
  }

  if (!body.garmentImage?.data || !body.garmentImage?.mimeType) {
    throw new AppError("Upload a garment image.", 400, "MISSING_GARMENT_IMAGE");
  }

  if (!Array.isArray(body.selectedPoses) || body.selectedPoses.length === 0) {
    throw new AppError("Select at least one pose.", 400, "MISSING_POSES");
  }

  const allowedImageSizes = new Set(["1K", "2K", "4K"]);
  if (!body.imageSize || !allowedImageSizes.has(body.imageSize)) {
    throw new AppError("imageSize must be one of: 1K, 2K, 4K.", 400, "INVALID_IMAGE_SIZE");
  }

  const allowedAspectRatios = new Set(["3:4", "1:1", "16:9", "9:16"]);
  if (!body.aspectRatio || !allowedAspectRatios.has(body.aspectRatio)) {
    throw new AppError("aspectRatio must be one of: 3:4, 1:1, 16:9, 9:16.", 400, "INVALID_ASPECT_RATIO");
  }
}

export const assertServerConfiguration = (): void => {
  const project =
    process.env.GOOGLE_CLOUD_PROJECT ||
    process.env.GCLOUD_PROJECT ||
    process.env.VERTEX_AI_PROJECT_ID;

  if (!project) {
    throw new AppError(
      "Missing Google Cloud project configuration. Set GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or VERTEX_AI_PROJECT_ID.",
      500,
      "MISSING_PROJECT_CONFIG",
    );
  }

  if (!process.env.GOOGLE_APPLICATION_CREDENTIALS && !process.env.GOOGLE_SERVICE_ACCOUNT_BASE64) {
    throw new AppError(
      "Missing Google credentials. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_BASE64.",
      500,
      "MISSING_CREDENTIALS",
    );
  }
};

const createClient = (): GoogleGenAI => {
  const project =
    process.env.GOOGLE_CLOUD_PROJECT ||
    process.env.GCLOUD_PROJECT ||
    process.env.VERTEX_AI_PROJECT_ID;

  const location =
    process.env.GOOGLE_CLOUD_LOCATION ||
    process.env.VERTEX_AI_LOCATION ||
    "global";

  if (!project) {
    throw new AppError(
      "Missing Google Cloud project configuration. Set GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or VERTEX_AI_PROJECT_ID.",
      500,
      "MISSING_PROJECT_CONFIG",
    );
  }

  ensureServiceAccountFile();

  console.log(`  [client] Vertex AI project=${project} location=${location} model=${MODEL_NAME}`);
  return new GoogleGenAI({ vertexai: true, project, location });
};

const generateWithRetry = async (
  ai: GoogleGenAI,
  model: string,
  contents: unknown,
  config: unknown,
  maxRetries = 2,
): Promise<unknown> => {
  let lastError: unknown;

  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await ai.models.generateContent({ model, contents: contents as never, config: config as never });
    } catch (error: unknown) {
      lastError = error;
      const err = error as Record<string, unknown>;
      const msg = String(err.message || "").toLowerCase();
      const status = err.status as number;
      const isRetryable =
        [429, 500, 503, 504].includes(status) ||
        msg.includes("overloaded") ||
        msg.includes("unavailable");

      if (isRetryable && i < maxRetries) {
        await delay(Math.pow(2, i) * 4000 + Math.random() * 1000);
        continue;
      }
      throw error;
    }
  }

  throw lastError;
};

export async function* streamPhotoshootImages(
  body: GenerateRequestBody,
): AsyncGenerator<GeneratedImagePayload> {
  validateGenerateRequest(body);

  const ai = createClient();
  const inputParts: unknown[] = [imagePart(body.garmentImage)];
  let characterIdentityPrompt: string;
  let environmentPrompt = body.backgroundPrompt;
  let designPrompt = "";
  let stylingPrompt = "";
  let dupattaPrompt = "";
  let blousePrompt = "";
  let backViewPrompt = "";

  if (body.blouseImage) {
    inputParts.push(imagePart(body.blouseImage));
    blousePrompt = "[STRICT BLOUSE/TOP LOCK]: Use the provided Blouse Reference Image for the exact color, pattern, cut, and fabric of the blouse or top.";
  }
  if (body.dupattaImage) {
    inputParts.push(imagePart(body.dupattaImage));
    dupattaPrompt = "[STRICT DUPATTA LOCK]: Use the provided Dupatta Reference Image for the exact color, pattern, border, and fabric.";
  }
  if (body.backViewImage) {
    inputParts.push(imagePart(body.backViewImage));
    backViewPrompt = "[BACK VIEW REFERENCE]: Use the provided Back View Reference Image to accurately render the back design and neckline.";
  }
  if (body.modelImage) {
    inputParts.push(imagePart(body.modelImage));
    characterIdentityPrompt = `REPLICATE THE PERSON IN THE CHARACTER REFERENCE IMAGE EXACTLY. Face shape, skin tone, and features must be identical. Notes: ${body.modelAttributes}.`;
  } else {
    characterIdentityPrompt = `${body.modelAttributes}. Consistent Indian model features with realistic skin textures.`;
  }
  if (body.backgroundImage) {
    inputParts.push(imagePart(body.backgroundImage));
    environmentPrompt = "PLACE THE MODEL IN THE EXACT ENVIRONMENT SHOWN IN THE BACKGROUND IMAGE.";
  }
  if (body.designDetailImage) {
    inputParts.push(imagePart(body.designDetailImage));
    designPrompt = "[MICROSCOPIC DESIGN OVERRIDE]: Use the provided Design Reference Image for all embroidery, handwork, and intricate patterns.";
  }
  if (body.stylingImage) {
    inputParts.push(imagePart(body.stylingImage));
    stylingPrompt = "[STYLE & POSE REFERENCE]: Use the provided Styling Reference Image for the overall aesthetic and lighting mood.";
  }

  const sessionLockProtocol = `
  [ULTRA-HIGH FIDELITY TEXTILE PROTOCOL]
  - MICROSCOPIC TEXTURE LOCK: Render the specific weave of the fabric with sub-pixel clarity.
  - SEAM & STITCHING REALISM: Include realistic seam puckering, tension at the stitch lines, and 3D depth for every visible thread.
  - EMBROIDERY DEPTH: ${designPrompt || "Preserve embroidery from garment source with extreme 3D detail."}
  - FABRIC INTEGRITY: Replicate the subtle fabric grain, weave, and physical weight including natural drapes and slight stress-wrinkles.
  - COLOR FIDELITY: Maintain the exact color hex, saturation, and subtle gradients of the original garment fabric.
  ${blousePrompt ? `- BLOUSE: ${blousePrompt}` : ""}
  ${dupattaPrompt ? `- DUPATTA: ${dupattaPrompt}` : ""}
  ${backViewPrompt ? `- BACK VIEW: ${backViewPrompt}` : ""}
  - CHARACTER: ${characterIdentityPrompt}
  - LOCATION: ${environmentPrompt}
  - STYLE: ${stylingPrompt || "Professional high-end fashion editorial style."}
  - CAMERA: Phase One XF, 100mm Macro Lens, f/11, ISO 50, professional studio high-CRI lighting.
  `;

  const generationConfig = {
    maxOutputTokens: 32768,
    temperature: 1,
    topP: 0.95,
    responseModalities: ["TEXT", "IMAGE"],
    imageConfig: {
      aspectRatio: body.aspectRatio,
      imageSize: body.imageSize,
      outputMimeType: "image/png",
    },
    safetySettings: [
      { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "OFF" },
      { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "OFF" },
      { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "OFF" },
      { category: "HARM_CATEGORY_HARASSMENT", threshold: "OFF" },
    ],
  };

  let firstGeneratedImagePart: unknown = null;

  for (const pose of body.selectedPoses) {
    const currentInputParts = [...inputParts];
    let consistencyPrompt = "";

    if (firstGeneratedImagePart) {
      currentInputParts.push(firstGeneratedImagePart);
      consistencyPrompt = `
      [STRICT CONSISTENCY LOCK]:
      1. EXACT SAME BACKGROUND: You MUST use the exact same background/location/environment as shown in the provided generated reference image.
      2. EXACT SAME MODEL LOOK: The model's face, hairstyle, makeup, and skin tone MUST be identical to the provided generated reference image.
      3. Only change the pose to match the [REQUIRED POSE].
      `;
    }

    const fullPrompt = `
    ${sessionLockProtocol}
    [REQUIRED POSE]: ${pose.prompt}
    ${consistencyPrompt}
    [FINAL MANDATE]: Replicate the reference garment's microscopic details with 100% accuracy. The output must look like a real, un-edited high-resolution photograph.
    `;

    const response = await generateWithRetry(
      ai,
      MODEL_NAME,
      [{ role: "user", parts: [...currentInputParts, { text: fullPrompt }] }],
      generationConfig,
    ) as Record<string, unknown>;

    const candidates = response.candidates as Array<Record<string, unknown>>;

    console.log(`  [${pose.title}] candidates count: ${candidates?.length}`);
    console.log(`  [${pose.title}] finishReason: ${candidates?.[0]?.finishReason}`);
    console.log(`  [${pose.title}] safetyRatings:`, JSON.stringify(candidates?.[0]?.safetyRatings ?? []));

    if (!candidates?.[0]?.content) {
      console.error(`  [${pose.title}] full response:`, JSON.stringify(response, null, 2));
      throw new Error(`Empty response for pose: ${pose.title}`);
    }

    const parts = (candidates[0].content as Record<string, unknown>).parts as Array<Record<string, unknown>>;
    console.log(`  [${pose.title}] parts count: ${parts?.length}, types: ${parts?.map(p => Object.keys(p).join('+')).join(', ')}`);

    const generatedPart = parts?.find((p) => p.inlineData);

    if (!generatedPart?.inlineData) {
      console.error(`  [${pose.title}] parts detail:`, JSON.stringify(parts?.map(p => ({ keys: Object.keys(p), text: (p.text as string)?.slice(0, 200) })), null, 2));
      throw new Error(`No image data for pose: ${pose.title}`);
    }

    const inlineData = generatedPart.inlineData as { mimeType: string; data: string };
    const mimeType = inlineData.mimeType || "image/png";
    const imageData = inlineData.data;

    // Update consistency reference BEFORE yielding so we can release data after
    if (!firstGeneratedImagePart) {
      firstGeneratedImagePart = { inlineData: { data: imageData, mimeType } };
    }

    yield {
      poseTitle: pose.title,
      url: `data:${mimeType};base64,${imageData}`,
    };

    // Explicitly release the large base64 payload from the response object
    // so the GC can reclaim it before the next iteration.
    (generatedPart.inlineData as Record<string, unknown>).data = null;
    (response as Record<string, unknown>).candidates = null;

    if (body.selectedPoses.indexOf(pose) < body.selectedPoses.length - 1) {
      await delay(1500);
    }
  }
}
