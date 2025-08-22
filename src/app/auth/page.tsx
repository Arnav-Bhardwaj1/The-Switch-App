// UploadExample.tsx
"use client";

import {
  ImageKitAbortError,
  ImageKitInvalidRequestError,
  ImageKitServerError,
  ImageKitUploadNetworkError,
  upload,
} from "@imagekit/next";
import { useRef, useState } from "react";

export default function UploadExample() {
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch upload auth params from server
  async function getAuthParams() {
    const res = await fetch("/api/imagekit-auth");
    if (!res.ok) throw new Error("Failed to get auth params");
    return res.json();
  }

  const handleUpload = async () => {
    if (!fileInputRef.current?.files?.length) {
      alert("Please select a file to upload");
      return;
    }
    const file = fileInputRef.current.files[0];
    let authParams;
    try {
      authParams = await getAuthParams();
    } catch (e) {
      console.error("Auth error", e);
      return;
    }

    try {
      const response = await upload({
        file,
        fileName: file.name,
        publicKey: authParams.publicKey,
        token: authParams.token,
        expire: authParams.expire,
        signature: authParams.signature,
        onProgress: (e) => setProgress((e.loaded / e.total) * 100),
        abortSignal: new AbortController().signal,
      });
      console.log("Upload success:", response);
      alert(`File uploaded! URL: ${response.url}`);
    } catch (error) {
      if (error instanceof ImageKitAbortError) {
        console.error("Upload aborted:", error.reason);
      } else if (error instanceof ImageKitInvalidRequestError) {
        console.error("Invalid request:", error.message);
      } else if (error instanceof ImageKitUploadNetworkError) {
        console.error("Network error:", error.message);
      } else if (error instanceof ImageKitServerError) {
        console.error("Server error:", error.message);
      } else {
        console.error("Unknown error:", error);
      }
    }
  };

  return (
    <div>
      <input type="file" ref={fileInputRef} />
      <button onClick={handleUpload}>Upload File</button>
      <div>Progress: {progress.toFixed(2)}%</div>
    </div>
  );
}
