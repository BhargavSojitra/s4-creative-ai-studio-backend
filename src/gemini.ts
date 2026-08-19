import { GoogleGenAI } from "@google/genai";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

const MODEL_NAME =
  process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image-preview";

type Pose = { id: string; title: string; prompt: string; category?: string };

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
  unstitchedSleeveImage?: InlineImagePayload | null;
  unstitchedCloseupImage?: InlineImagePayload | null;
  dummyShootImage?: InlineImagePayload | null;
  expressionPrompt?: string;
  necklinePrompt?: string;
  sleevePrompt?: string;
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

  const expressionPrompt = body.expressionPrompt || "";
  const necklinePrompt = body.necklinePrompt || "";
  const sleevePrompt = body.sleevePrompt || "";

  const inputParts: unknown[] = [imagePart(body.garmentImage)];
  const imageDescriptions: string[] = [
    "Input Image 1: The original uploaded garment. This is the absolute master reference for the fabric, colors, pattern, print, designs, and texture. Do NOT change or add any embroidery/patterns not visible here.",
  ];
  let imageIndex = 2;

  let characterIdentityPrompt: string;
  let environmentPrompt = body.backgroundPrompt;
  let designPrompt = "";
  let stylingPrompt = "";
  let dupattaPrompt = "";
  let blousePrompt = "";
  let backViewPrompt = "";
  let unstitchedPrompt = "";
  let dummyShootPrompt = "";

  if (body.dummyShootImage) {
    inputParts.push(imagePart(body.dummyShootImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Dummy Shoot Reference showing desired layout and bottom styling.`);
    dummyShootPrompt = `[DUMMY SHOOT REFERENCE - Input Image ${imageIndex}]: Use this image to understand the required garment structure, specifically the pant/bottom style, the way the fabric drapes, and the overall outfit silhouette. The final garment construction MUST reflect this reference.`;
    imageIndex++;
  }
  if (body.unstitchedSleeveImage) {
    inputParts.push(imagePart(body.unstitchedSleeveImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Unstitched Sleeve fabrics reference.`);
    unstitchedPrompt += `[UNSTITCHED SLEEVE REFERENCE - Input Image ${imageIndex}]: Use the provided Sleeve Fabric Image to render the sleeves of the outfit. The sleeves should be stitched according to this fabric's pattern and texture. `;
    imageIndex++;
  }
  if (body.unstitchedCloseupImage) {
    inputParts.push(imagePart(body.unstitchedCloseupImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Close-up photo showing cloth weave, print, and exact print details.`);
    unstitchedPrompt += `[UNSTITCHED CLOSE-UP REFERENCE - Input Image ${imageIndex}]: Refer to the Close-up Image for CRYSTAL CLEAR fabric quality, pattern, texture, print quality, and original details. Replicate the original design features with absolute accuracy and extreme sharpness across the entire stitched garment. `;
    imageIndex++;
  }
  if (body.blouseImage) {
    inputParts.push(imagePart(body.blouseImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Blouse Reference Image.`);
    blousePrompt = `[STRICT BLOUSE/TOP LOCK - Input Image ${imageIndex}]: Use the provided Blouse Reference Image for the exact color, pattern, cut, and fabric of the blouse or top. The blouse in the generated image MUST perfectly match this reference image.`;
    imageIndex++;
  }
  if (body.dupattaImage) {
    inputParts.push(imagePart(body.dupattaImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Dupatta Reference Image.`);
    dupattaPrompt = `[STRICT DUPATTA LOCK - Input Image ${imageIndex}]: Use the provided Dupatta Reference Image for the exact color, pattern, border, and fabric of the dupatta. The dupatta in the generated image MUST perfectly match this reference image. Do not hallucinate a different dupatta.`;
    imageIndex++;
  }
  if (body.backViewImage) {
    inputParts.push(imagePart(body.backViewImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Back View Reference Image.`);
    backViewPrompt = `[BACK VIEW REFERENCE - Input Image ${imageIndex}]: Use the provided Back View Reference Image to accurately render the back design, neckline, and details of the garment. Ensure the back of the garment matches this reference exactly when the model is turned around.`;
    imageIndex++;
  }
  if (body.modelImage) {
    inputParts.push(imagePart(body.modelImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Model Character Reference.`);
    characterIdentityPrompt = `REPLICATE THE PERSON IN THE CHARACTER REFERENCE IMAGE [Input Image ${imageIndex}] EXACTLY. Face shape, skin tone, and features must be identical. ${expressionPrompt ? `EXPRESSION: ${expressionPrompt}.` : ""} Notes: ${body.modelAttributes}.`;
    imageIndex++;
  } else {
    characterIdentityPrompt = `${body.modelAttributes}. Consistent Indian model features with realistic skin textures. ${expressionPrompt ? `EXPRESSION: ${expressionPrompt}.` : ""}`;
  }
  if (body.backgroundImage) {
    inputParts.push(imagePart(body.backgroundImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Background Scene Reference.`);
    environmentPrompt = `PLACE THE MODEL IN THE EXACT ENVIRONMENT SHOWN IN THE BACKGROUND IMAGE [Input Image ${imageIndex}].`;
    imageIndex++;
  }
  if (body.designDetailImage) {
    inputParts.push(imagePart(body.designDetailImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Direct Design/Pattern/Embroidery Closeup Reference.`);
    designPrompt = `[MICROSCOPIC DESIGN OVERRIDE - Input Image ${imageIndex}]: Use the provided Design Reference Image for all print, pattern, embroidery, or handwork. Replicate the specific patterns with extreme sharpness and clarity.`;
    imageIndex++;
  }
  if (body.stylingImage) {
    inputParts.push(imagePart(body.stylingImage));
    imageDescriptions.push(`Input Image ${imageIndex}: Pose & Aesthetic Styling Reference.`);
    stylingPrompt = `[STYLE & POSE REFERENCE - Input Image ${imageIndex}]: Use the provided Styling Reference Image for the overall aesthetic, lighting mood, and model's pose/attitude. Replicate the vibe and composition style while maintaining the garment and model identity.`;
    imageIndex++;
  }

  const hasLongShot = body.selectedPoses.some((p) => p.category === "Long Shot");

  const sessionLockProtocol = `
    [INPUT IMAGE INDEXING & ROLES]
    ${imageDescriptions.map((desc) => "- " + desc).join("\n")}

    [ULTRA-HIGH FIDELITY TEXTILE & CRAFT PROTOCOL]
    - MICROSCOPIC TEXTURE LOCK: Render the absolute highest resolution fabric weave, fiber details, and precise color tone of Input Image 1. Replicate it with extreme pixel-perfect sharpness and crystal-clear clarity.
    - CRAFT & EMBROIDERY PRECISION: Ensure that every piece of garment work (including hand embroidery, zari threadwork, delicate sequin arrangements, beadwork, mirror-work, lace detailing, and fine borders) is hyper-focused, completely sharp, and deeply highlighted. No parts of the embroidery or pattern should be blurry, muddy, or smoothed out.
    - SEAM & STITCHING REALISM: Render realistic seam puckering, thread tension at the stitch lines, and 3D depth for every single visible stitch, piping line, and border trim with maximum definition.
    - GRAPHIC & DETAIL ACCURACY: ${designPrompt || "Do NOT add any extra, new, or unrequested embroidery, lace, border, beadwork, or embellishments. Replicate ONLY the exact print, design, solid color, embroidery, or pattern present in the original uploaded garment image (Input Image 1). If Input Image 1 is solid/plain, keep the generated garment 100% solid/plain without adding any prints or embroidery."}
    - NO DESIGN DEVIATION: Replicate all print shapes, floral motifs, geometric stripes, and borders from Input Image 1 exactly. Maintain identical scaling, spacing, and placement without distorting or randomizing the fabric patterns.
    - FABRIC INTEGRITY: Retain the natural fabric characteristics (such as the raw slub of silk, the delicate texture of georgette, or the crisp lines of linen) with high-definition, tactile realism.
    - COLOR FIDELITY: Replicate the precise color tones, deep pigments, and subtle dye variations of the original garment fabric with zero alteration.
    ${blousePrompt ? `- BLOUSE: ${blousePrompt}` : ''}
    ${dupattaPrompt ? `- DUPATTA: ${dupattaPrompt}` : ''}
    ${backViewPrompt ? `- BACK VIEW: ${backViewPrompt}` : ''}
    ${unstitchedPrompt ? `- UNSTITCHED COMPONENTS: ${unstitchedPrompt}` : ''}
    ${dummyShootPrompt ? `- DUMMY REFERENCE: ${dummyShootPrompt}` : ''}
    ${necklinePrompt ? `- NECKLINE STYLE: ${necklinePrompt}` : ''}
    ${sleevePrompt ? `- SLEEVE STYLE: ${sleevePrompt}` : ''}
    - SLEEVE INTEGRATION: The sleeves MUST be rendered as an inseparable part of the main garment, with matching fabric, print, and seamless stitching. If a sleeve fabric is provided, blend it perfectly into the shoulder and armhole of the top.
    - GARMENT SHAPE & DRAPE: The garment MUST naturally take the shape of the model's body and the chosen pant type. Ensure realistic fabric folds, shadows, and drapes that reflect the underlying physical structure of the model and bottoms.
    - CHARACTER: ${characterIdentityPrompt}
    - LOCATION: ${environmentPrompt}
    - STYLE: ${stylingPrompt || "Professional high-end fashion editorial style."}
    - COMPOSITION: ${hasLongShot ? "Varies based on pose. For Long Shots, use a wide-angle cinematic composition capturing the model within the environment. For standard poses, the model should be the central focus." : "Full-body or mid-shot framing. The model MUST be the dominant central focus, filling 85-95% of the vertical frame height. Zero space above the head or below the feet in full-body shots. Absolutely no wide-angle 'small person' shots unless explicitly requested by a Long Shot pose."}
    - HYPER-REALISM & CAMERA PROTOCOL:
      1. NO AI SMOOTHING: Absolutely zero plastic skin, zero generic AI smoothing, and zero synthetic "airbrushed" filter artifacts. Show authentic, pore-level skin texture, fine hair, and real light reflections.
      2. CAMERA AND LENS: Capture using a high-end medium format camera (Phase One XF IQ4 150MP or Hasselblad H6D-100c) with a premium macro-capable prime lens (100mm f/8).
      3. EXTREME DEPTH OF FIELD: Maintain deep sharp focus across the entire model and garment. Every stitch, zari wire, and bead from shoulder to hem must remain pin-sharp and crystal clear.
      4. STUDIO LIGHTING: Render with high-CRI professional commercial studio lighting (crisp key light, subtle fill, and crisp rim highlights) to beautifully accentuate the embroidery texture, metallic shine, and fabric sheen.
    `;

  const generationConfig = {
    maxOutputTokens: 32768,
    temperature: 0.4,
    topP: 0.8,
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
            1. EXACT SAME BACKGROUND: You MUST use the exact same background/location/environment as shown in the provided generated reference image. Do not change the setting.
            2. EXACT SAME MODEL LOOK: The model's face, facial features, hairstyle, makeup, and skin tone MUST be identical to the provided generated reference image.
            3. Only change the pose/movement to match the [REQUIRED POSE].
            `;
    }

    const fullPrompt = `
        ${sessionLockProtocol}
        [REQUIRED POSE]: ${pose.prompt}
        ${consistencyPrompt}
        [FINAL MANDATE]: Replicate the reference garment from Input Image 1's features, print, design, and colors with 100% accuracy. Ensure all garment work, stitch lines, prints, and embroideries are extremely sharp, high-definition, and look like a real physical photoshoot shot with a high-end commercial DSLR. Do NOT add any extra, new, or unrequested embroidery, thread-work, or patterns. Maintain every unique textile print and design feature with pristine clarity.
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
