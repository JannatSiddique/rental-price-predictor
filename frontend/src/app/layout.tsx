import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Navbar from "@/components/Navbar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Rental Price Predictor",
  description: "Estimate monthly rent based on property details",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased scroll-smooth`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-black text-white">
        <Navbar />
        <div className="flex flex-col min-h-screen">
          {children}
          <footer className="border-t border-white/10 bg-black">
            <div className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 sm:grid-cols-3 gap-8">
              <div>
                <h3 className="text-lg font-bold bg-gradient-to-r from-purple-400 to-white bg-clip-text text-transparent mb-3">
                  RentPredict
                </h3>
                <p className="text-sm text-gray-500 leading-relaxed">
                  AI-powered rental price estimation for properties across Pakistan. Fast, accurate, and reliable.
                </p>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Quick Links</h4>
                <ul className="space-y-2 text-sm text-gray-500">
                  <li><a href="#hero" className="hover:text-purple-400 transition">Home</a></li>
                  <li><a href="#predict" className="hover:text-purple-400 transition">Predict</a></li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Tech Stack</h4>
                <ul className="space-y-2 text-sm text-gray-500">
                  <li>Next.js &middot; Tailwind CSS</li>
                  <li>FastAPI &middot; Python</li>
                  <li>Scikit-learn &middot; Streamlit</li>
                </ul>
              </div>
            </div>
            <div className="border-t border-white/5 py-6">
              <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-600">
                <span>&copy; {new Date().getFullYear()} Rental Price Predictor. All rights reserved.</span>
                <span>Powered by Machine Learning</span>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
