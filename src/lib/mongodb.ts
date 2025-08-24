// import mongoose from "mongoose"

// const MONGODB_URI = process.env.MONGODB_URI as string

// if (!MONGODB_URI) {
//   throw new Error("⚠️ Please define the MONGODB_URI environment variable")
// }

// const cached = (global as { mongoose?: { conn: mongoose.Connection | null; promise: Promise<mongoose.Connection> | null } }).mongoose || { conn: null, promise: null }

// export async function connectToDB() {
//   if (cached.conn) return cached.conn

//   if (!cached.promise) {
//     cached.promise = mongoose.connect(MONGODB_URI, {
//       dbName: "switch",
//       bufferCommands: false,
//     })
//   }

//   cached.conn = await cached.promise
//   return cached.conn
// }



// import mongoose from "mongoose"

// const MONGODB_URI = process.env.MONGODB_URI as string

// if (!MONGODB_URI) {
//   throw new Error("⚠️ Please define the MONGODB_URI environment variable")
// }

// /**
//  * Global is used here to maintain a cached connection across hot reloads
//  * in development. This prevents connections from growing exponentially
//  * during API Route usage.
//  */
// let cached = (global as any).mongoose

// if (!cached) {
//   cached = (global as any).mongoose = { conn: null, promise: null }
// }

// export async function connectToDB() {
//   if (cached.conn) {
//     return cached.conn
//   }

//   if (!cached.promise) {
//     const opts = {
//       dbName: "switch",
//       bufferCommands: false,
//     }

//     // The connect() promise resolves to the mongoose instance.
//     // We chain .then() to get a new promise that resolves to the connection object.
//     cached.promise = mongoose.connect(MONGODB_URI, opts).then((mongoose) => {
//       return mongoose.connection
//     })
//   }

//   try {
//     cached.conn = await cached.promise
//   } catch (e) {
//     // If the connection fails, reset the promise so we can try again.
//     cached.promise = null
//     throw e
//   }
  
//   return cached.conn
// }


import mongoose, { Connection } from "mongoose";

const MONGODB_URI = process.env.MONGODB_URI as string;

if (!MONGODB_URI) {
  throw new Error("⚠️ Please define the MONGODB_URI environment variable");
}

// Define an interface for our cached mongoose object to avoid using 'any'.
interface MongooseCache {
  conn: Connection | null;
  promise: Promise<Connection> | null;
}

// Augment the NodeJS global type to include our 'mongoose' property.
// This tells TypeScript that 'global.mongoose' can exist and what its shape is.
declare global {
  var mongoose: MongooseCache;
}

/**
 * Global is used here to maintain a cached connection across hot reloads
 * in development. This prevents connections from growing exponentially
 * during API Route usage.
 */
let cached: MongooseCache = global.mongoose;

if (!cached) {
  cached = global.mongoose = { conn: null, promise: null };
}

export async function connectToDB() {
  if (cached.conn) {
    return cached.conn;
  }

  if (!cached.promise) {
    const opts = {
      dbName: "switch",
      bufferCommands: false,
    };

    // The connect() promise resolves to the mongoose instance.
    // We chain .then() to get a new promise that resolves to the connection object.
    cached.promise = mongoose.connect(MONGODB_URI, opts).then((mongoose) => {
      return mongoose.connection;
    });
  }

  try {
    cached.conn = await cached.promise;
  } catch (e) {
    // If the connection fails, reset the promise so we can try again.
    cached.promise = null;
    throw e;
  }
  
  return cached.conn;
}
