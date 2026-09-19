import { useMutation } from "@tanstack/react-query";
import { analyzeImage } from "../api/image";
import { analyzeText, uploadText } from "../api/text";
import { uploadVideo } from "../api/video";

export function useDetection() {
  return {
    text: useMutation({ mutationFn: analyzeText }),
    textUpload: useMutation({ mutationFn: uploadText }),
    image: useMutation({ mutationFn: analyzeImage }),
    video: useMutation({ mutationFn: uploadVideo })
  };
}
