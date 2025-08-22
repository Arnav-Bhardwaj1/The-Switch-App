import mongoose, { Schema, model, models } from 'mongoose';

const CommentSchema = new Schema({
  post: { type: mongoose.Schema.Types.ObjectId, ref: 'Post', required: true },
  author: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  text: { type: String, required: true },
}, { timestamps: true });

export const Comment = models.Comment || model('Comment', CommentSchema);