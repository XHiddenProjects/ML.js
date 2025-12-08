// PixelDetector
import { PixelDetection } from "./classes/artificial_intelligence.class.js";
const detect = new PixelDetection({
    targetSize: 256,
    resizeMode: 'contain',
    grayscale: true,
    videoFrameSamples: 12,
    weights: {mse: 1, psnr: 1, ssim: 2, hist: 1, phash: 2},
    maxPSNR: 60
});

const x = await detect.compare(
    'https://imgs.search.brave.com/h_LgG9E14kkNkyAZ-Y128Roa8zN6ZDUjsq1PlJ88pQc/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudmVjdGVlenku/Y29tL3N5c3RlbS9y/ZXNvdXJjZXMvdGh1/bWJuYWlscy8wMzkv/MjEzLzkzNi9zbWFs/bC9haS1nZW5lcmF0/ZWQtYWJzdHJhY3Qt/d2F0ZXItYnViYmxl/cy1jb2xvcmZ1bC1i/YWNrZ3JvdW5kLWRl/c2lnbi1pbWFnZXMt/cGhvdG8uanBn',
    'https://imgs.search.brave.com/vjAzuZuJK0JqVrFE9OWV3LF5VO4bvwqBmS_WQaOSYLI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZWpvdXJuYWwub3Jn/L3dwLWNvbnRlbnQv/dXBsb2Fkcy8yMDI1/LzEwL0ltYWdlcy1m/b3ItV2ViX1BhZ2Vf/MDQ4X0ltYWdlXzAw/MDEtMS1zY2FsZWQu/anBn'
);

console.log(x.similarity);