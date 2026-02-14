import React from 'react'

function Sidebar() {
  return (
    <div className="w-64 bg-slate-900 text-white p-6">
      <h2 className="text-2xl font-bold mb-10">Dashboard</h2>

      <ul className="space-y-4 text-gray-300">
        <li className="hover:text-white cursor-pointer">Dashboard</li>
        <li className="hover:text-white cursor-pointer">Visitors</li>
        <li className="hover:text-white cursor-pointer">Employees</li>
        <li className="hover:text-white cursor-pointer">Reports</li>
      </ul>
    </div>
  )
}

export default Sidebar

