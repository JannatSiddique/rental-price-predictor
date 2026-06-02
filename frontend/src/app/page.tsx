import PredictionForm from "@/components/PredictionForm";

export default function Home() {
  return (
    <>
      <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background pattern */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wMyI+PHBhdGggZD0iTTM2IDM0djItSDI0di0yaDEyek0zNiAyNHYySDI0di0yaDEyeiIvPjwvZz48L2c+PC9zdmc+')] opacity-50" />

        {/* Gradient overlays */}
        <div className="absolute inset-0 bg-gradient-to-b from-black via-black/95 to-black" />
        <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-purple-600/20 rounded-full blur-[150px] pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-purple-800/15 rounded-full blur-[120px] pointer-events-none" />

        {/* Grid lines */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
            backgroundSize: '60px 60px',
          }}
        />

        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-purple-500/30 bg-purple-950/40 text-purple-300 text-xs sm:text-sm font-medium mb-8 backdrop-blur-sm">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-purple-400 glow-pulse" />
            AI-Powered Price Estimation
          </div>

          <h1 className="text-5xl sm:text-6xl lg:text-8xl font-bold tracking-tight leading-tight">
            <span className="bg-gradient-to-r from-white via-purple-100 to-purple-300 bg-clip-text text-transparent">
              Know Your
            </span>
            <br />
            <span className="bg-gradient-to-r from-purple-400 via-purple-200 to-white bg-clip-text text-transparent">
              Property&apos;s Worth
            </span>
          </h1>

          <p className="mt-6 text-lg sm:text-xl text-gray-500 max-w-2xl mx-auto leading-relaxed">
            Get an accurate monthly rent estimate for any property in Pakistan.
            Just enter the details and let our ML model do the rest.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <a
              href="#predict"
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-purple-600 to-purple-500 text-white font-semibold hover:from-purple-500 hover:to-purple-400 transition shadow-lg shadow-purple-500/25"
            >
              Predict Now
            </a>
            <a
              href="#predict"
              className="px-8 py-3.5 rounded-xl border border-white/10 text-gray-300 font-medium hover:bg-white/5 hover:text-white transition"
            >
              Learn More
            </a>
          </div>

          <div className="mt-16 grid grid-cols-3 gap-8 max-w-lg mx-auto">
            {[
              { value: "80+", label: "Locations" },
              { value: "ML", label: "Powered" },
              { value: "PKR", label: "Estimates" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-2xl sm:text-3xl font-bold text-white">{stat.value}</div>
                <div className="text-xs sm:text-sm text-gray-600 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="predict" className="relative py-20 sm:py-28 px-4">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-purple-950/5 to-black pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-purple-600/5 rounded-full blur-[150px] pointer-events-none" />

        <div className="relative max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold">
              <span className="bg-gradient-to-r from-white to-purple-300 bg-clip-text text-transparent">
                Estimate Your Rent
              </span>
            </h2>
            <p className="mt-3 text-gray-500 max-w-lg mx-auto">
              Fill in the property details below and get an instant AI-powered prediction.
            </p>
          </div>

          <div className="max-w-lg mx-auto">
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-2xl p-6 sm:p-8 shadow-2xl">
              <PredictionForm />
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
