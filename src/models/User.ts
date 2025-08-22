import mongoose, { Schema, model, models } from 'mongoose';

const UserSchema = new Schema({
  name: { type: String, required: true },
  username: { type: String, unique: true, required: true },
  email: { type: String, unique: true, required: true },
  password: { type: String, required: true },
  gender: { type: String, enum: ['male', 'female', 'trans', 'other'], required: true },
  interests: [{ type: String }],
  bio: { type: String, default: '' },
  profileImage: { type: String, default: '' },

  followers: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
  following: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],

  isVerified: {
    type: Boolean,
    default: false,
  },
  otp: {
    type: String,
  },
  otpExpires: {
    type: Date,
  },


  // ✅ User's posts
  posts: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Post' }],
  savedPosts: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Post' }],

  conversations: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Conversation' }],


}, { timestamps: true });

export const User = models.User || model('User', UserSchema);
