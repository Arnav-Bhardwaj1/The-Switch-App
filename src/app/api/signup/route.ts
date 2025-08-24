import { connectToDB } from '@/lib/mongodb';
import { User } from '@/models/User';
import bcrypt from 'bcryptjs';
import nodemailer from 'nodemailer';

// Nodemailer transporter setup
// IMPORTANT: Use environment variables for your email credentials.
const transporter = nodemailer.createTransport({
  service: 'gmail', // Or your email provider
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

export async function POST(req: Request) {
  try {
    const { name, username, email, password, gender, bio, profileImage } = await req.json();

    // --- Basic Validation ---
    if (!name || !username || !email || !password || !gender) {
      return new Response(JSON.stringify({ error: 'All required fields must be filled.' }), { status: 400 });
    }

    await connectToDB();

    // --- Check for existing user ---
    const existingUser = await User.findOne({ email });
    if (existingUser && existingUser.isVerified) {
      return new Response(JSON.stringify({ error: 'Email is already registered and verified.' }), { status: 400 });
    }

    const existingUsername = await User.findOne({ username });
    if (existingUsername) {
      return new Response(JSON.stringify({ error: 'Username is already taken.' }), { status: 400 });
    }

    // --- OTP Generation ---
    const otp = Math.floor(100000 + Math.random() * 900000).toString();
    const otpExpires = new Date(Date.now() + 10 * 60 * 1000); // OTP expires in 10 minutes

    // --- Password Hashing ---
    const hashedPassword = await bcrypt.hash(password, 10);

    // --- Create or Update User ---
    // If user exists but is not verified, update their info and OTP
    if (existingUser) {
      existingUser.name = name;
      existingUser.username = username;
      existingUser.password = hashedPassword;
      existingUser.gender = gender;
      existingUser.bio = bio || '';
      existingUser.profileImage = profileImage || '';
      existingUser.otp = otp;
      existingUser.otpExpires = otpExpires;
      await existingUser.save();
    } else {
      const newUser = new User({
        name,
        username,
        email,
        password: hashedPassword,
        gender,
        bio: bio || '',
        profileImage: profileImage || '',
        otp,
        otpExpires,
        isVerified: false,
      });
      await newUser.save();
    }


    // --- Send Verification Email ---
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Verify Your Email Address',
      html: `
<div style="font-family: Arial, sans-serif; color: #333;">
    <h2>Hello ${name},</h2>
    <p>Thanks for choosing <strong>Switch</strong> – Your Choice, Your Space!</p>
    <p>Please use the following One-Time Password (OTP) to verify your email address:</p>
    <p style="font-size: 24px; font-weight: bold; letter-spacing: 2px;">${otp}</p>
    <p>This OTP will expire in 10 minutes.</p>
    <p>If you did not request this, please ignore this email.</p>
  </div>      `,
    };

    await transporter.sendMail(mailOptions);

    return new Response(JSON.stringify({ message: 'Signup successful. Please check your email for the verification OTP.' }), { status: 201 });

  } catch (error) {
    console.error('[SIGNUP_ERROR]', error);
    return new Response(JSON.stringify({ error: 'Signup failed. Please try again later.' }), { status: 500 });
  }
}
