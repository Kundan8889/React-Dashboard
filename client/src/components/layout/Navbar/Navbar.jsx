import React from "react";

function Navbar() {
  return (
    <div className="bg-white shadow-sm px-8 py-4 flex justify-between items-center">
      <h1 className="text-xl font-semibold text-slate-700">
        Visitor Management
      </h1>
      <div className="bg-slate-100 px-4 py-2 rounded-full text-sm">Admin</div>
    </div>
  );
}

export default Navbar;
