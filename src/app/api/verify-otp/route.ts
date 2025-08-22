/*
File: /app/api/verify-otp/route.ts
Description: Verifies the OTP submitted by the user.
*/

import { connectToDB } from '@/lib/mongodb';
import { User } from '@/models/User';

export async function POST(req: Request) {
  try {
    const { email, otp } = await req.json();

    if (!email || !otp) {
      return new Response(JSON.stringify({ error: 'Email and OTP are required.' }), { status: 400 });
    }

    await connectToDB();

    const user = await User.findOne({ email });

    if (!user) {
      return new Response(JSON.stringify({ error: 'User not found.' }), { status: 404 });
    }

    if (user.isVerified) {
      return new Response(JSON.stringify({ message: 'Email is already verified.' }), { status: 200 });
    }

    // --- OTP Validation ---
    const isOtpValid = user.otp === otp;
    const isOtpExpired = user.otpExpires && user.otpExpires < new Date();

    if (isOtpExpired) {
      return new Response(JSON.stringify({ error: 'OTP has expired. Please sign up again to get a new one.' }), { status: 400 });
    }
    
    if (!isOtpValid) {
      return new Response(JSON.stringify({ error: 'Invalid OTP.' }), { status: 400 });
    }

    // --- Mark user as verified ---
    user.isVerified = true;
    user.otp = undefined; // Clear OTP fields
    user.otpExpires = undefined;
    await user.save();

    return new Response(JSON.stringify({ message: 'Email verified successfully!' }), { status: 200 });

  } catch (error) {
    console.error('[VERIFY_OTP_ERROR]', error);
    return new Response(JSON.stringify({ error: 'Verification failed. Please try again later.' }), { status: 500 });
  }
}
