import { FaUsers } from "react-icons/fa";

function StatCard({ title, value }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm p-6 flex justify-between items-center hover:shadow-lg hover:-translate-y-1 transition duration-300">
      <div>
        <p className="text-gray-500 text-sm">{title}</p>
        <h2 className="text-3xl font-bold mt-2">{value}</h2>
      </div>
      <FaUsers className="text-3xl text-slate-400" />
      
    </div>
  );
}
export default StatCard;
