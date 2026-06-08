import Sidebar from "@/components/shell/Sidebar";
import TopBar from "@/components/shell/TopBar";

export default function PlatformLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="app">
      <Sidebar />
      <div className="main">
        <TopBar />
        {children}
      </div>
    </div>
  );
}
