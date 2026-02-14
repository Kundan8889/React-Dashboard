import React from 'react'
import Dashboard from './pages/Dashboard'
import Layout from './components/layout/Layout'

const App = () => {
  return (
    <div className='flex min-h-screen bg-gray-100'>
     <Layout>
      <Dashboard/>
     </Layout>
    </div>
  )
}

export default App
