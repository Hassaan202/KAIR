import type { Metadata } from "next";
import "@/styles/globals.css";
import Providers from "./providers";

export const metadata: Metadata = {
  title: "SUPARCO · Super-Resolution Lab",
  description: "Satellite image super-resolution — preprocessing, training, and inference.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
