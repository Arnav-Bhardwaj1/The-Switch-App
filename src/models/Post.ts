import mongoose, { Schema, model, models } from 'mongoose';

const PostSchema = new Schema({
  author: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },

  text: { type: String, default: '' },

  photo: { type: String, default: '' },  // URL
  video: { type: String, default: '' },  // URL

  topics: [{ type: String }],  // e.g., ['sports', 'news', 'technology']
  tags: [{ type: String }],    // e.g., ['fun', 'morning', 'travel']

  space: { type: String, default: 'common' },

  likes: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
  comments: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Comment' }],

  savedBy: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],   // Users who saved this post
  sharedBy: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],  // Users who shared this post

}, { timestamps: true });

export const Post = models.Post || model('Post', PostSchema);
