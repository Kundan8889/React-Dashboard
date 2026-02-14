import Navbar from "./Navbar/Navbar"
import Sidebar from "./Sidebar/Sidebar"

function Layout({ children }) {
  return (
 <div className="flex-1 bg-gray-100">
  <Navbar />
  <div className="p-8 max-w-6xl mx-auto">
    {children}
  </div>
</div>
  )
}

export default Layout
