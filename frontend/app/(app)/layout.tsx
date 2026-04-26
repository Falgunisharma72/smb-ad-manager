import { Navbar } from "@/components/navbar";
import { HapticProvider } from "@/components/haptic-provider";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <HapticProvider />
      <Navbar />
      <main className="px-6 md:px-10 lg:px-16 py-12 md:py-16">{children}</main>
    </div>
  );
}
