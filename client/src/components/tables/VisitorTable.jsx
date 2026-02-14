import React, { useRef, useState } from "react"

function VisitorTable({ visitors }) {
 const [searchTerm, setSearchTerm] = useState("");
  const fileInputRef = useRef(null);

  const filteredVisitors = visitors.filter((v) =>
  v.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
  v.meetTo.toLowerCase().includes(searchTerm.toLowerCase())
);

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    console.log("Selected Video:", file);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl shadow-md p-6">
      
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-lg font-semibold">Recent Visitors</h2>

   <input
  type="text"
  placeholder="Search visitor..."
  value={searchTerm}
  onChange={(e) => setSearchTerm(e.target.value)}
  className="border border-slate-300 px-3 py-2 rounded-lg text-sm 
             focus:outline-none focus:ring-2 focus:ring-slate-400"
/>


        <button
          onClick={handleButtonClick}
          className="bg-slate-900 text-white px-4 py-2 rounded-lg hover:bg-slate-700 transition text-sm"
        >
          + Upload Video
        </button>
        <input
          type="file"
          accept="video/*"
          className="hidden"
          ref={fileInputRef}
          onChange={handleFileChange}
        />
      </div>

      <table className="w-full text-left">
        <thead className="text-sm text-slate-500 uppercase tracking-wide">
          <tr className="border-b">
            <th className="py-3">Name</th>
            <th className="py-3">Meet To</th>
            <th className="py-3">Status</th>
          </tr>
        </thead>

        <tbody>
           {filteredVisitors.map((v) => (
            <tr key={v.id} className="border-b hover:bg-gray-50">
              <td className="py-3">{v.name}</td>
              <td>{v.meetTo}</td>
              <td>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    v.status === "IN"
                      ? "bg-green-100 text-green-600"
                      : "bg-red-100 text-red-600"
                  }`}
                >
                  {v.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

    </div>
  );
}

export default VisitorTable;
