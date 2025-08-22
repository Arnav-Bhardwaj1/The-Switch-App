'use client'

import Link from "next/link"
import { motion } from "framer-motion"
import { useSession } from "next-auth/react"
import { Loader2 } from "lucide-react"

export default function Home() {
  // 1. Get the session status and data.
  // The 'status' can be 'loading', 'authenticated', or 'unauthenticated'.
  const { data: session, status } = useSession()

  return (
    <main className="min-h-screen bg-gradient-to-br from-indigo-100 to-white flex flex-col items-center justify-center text-center p-6">
      <motion.h1
        className="text-5xl font-bold mb-4 text-gray-800"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        Switch
      </motion.h1>
      
      {/* 2. Conditionally update the subtitle based on login status */}
      <motion.p 
        className="text-xl max-w-xl text-gray-700 mb-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        {status === 'authenticated' ? (
          <>Welcome back, <span className="font-semibold text-indigo-600">{session.user?.name || 'User'}</span>!</>
        ) : (
          <>
            “Your Space. Your Feed. Your Growth.” <br />
            A gender-safe, interest-focused social media platform for Gen Z.
          </>
        )}
      </motion.p>
      
      <div className="flex gap-4 h-12 items-center">
        {/* 3. Conditionally render the correct buttons */}

        {/* While loading, show a spinner to prevent UI flashing */}
        {status === "loading" && (
          <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
        )}

        {/* If user is logged IN, show the dashboard button */}
        {status === "authenticated" && (
          <Link href="/explore">
            <motion.button 
              className="px-8 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition shadow-lg hover:shadow-xl"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Explore
            </motion.button>
          </Link>
        )}

        {/* If user is logged OUT, show the signup/login buttons */}
        {status === "unauthenticated" && (
          <>
            <Link href="/auth/signup">
              <button className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition">
                Sign Up
              </button>
            </Link>
            <Link href="/auth/login">
              <button className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition">
                Log In
              </button>
            </Link>
          </>
        )}
      </div>
    </main>
  )
}