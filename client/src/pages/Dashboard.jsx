
import StatCard from "../components/cards/StatCard"
import VisitorTable from "../components/tables/VisitorTable"
import { stats, recentVisitors } from "../data/mockData"

function Dashboard() {
  return (
    <div className="space-y-6">
      
      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Total Visitors" value={stats.totalVisitors} />
        <StatCard title="Active Visitors" value={stats.activeVisitors} />
        <StatCard title="Total Employees" value={stats.totalEmployees} />
        <StatCard title="Exited Today" value={stats.exitedToday} />
      </div>

      {/* Table Section */}
      <VisitorTable visitors={recentVisitors} />

    </div>
  )
}

export default Dashboard
