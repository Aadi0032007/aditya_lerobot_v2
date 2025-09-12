# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:58:20 2025

@author: aadi
"""
from google import genai
from google.genai import types
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import torch
import io
import os
from dotenv import load_dotenv
import requests
from io import BytesIO
import random
import time

import dataclasses
import numpy as np
import base64
import math

import json

# Load environment variables from the .env file
load_dotenv() 

# Access your Gemini API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)
model_name = "gemini-2.5-flash-preview-05-20" # @param ["gemini-1.5-flash-latest","gemini-2.0-flash-lite","gemini-2.0-flash","gemini-2.5-flash-preview-05-20","gemini-2.5-pro-preview-06-05"] {"allow-input":true}
pro_model_name = "gemini-2.5-pro-preview-06-05"




def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor to PIL Image"""
    if len(tensor.shape) == 4:  # Remove batch dimension if present
        tensor = tensor[0]
    
    # Convert from CxHxW to HxWxC
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        img_array = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img_array = tensor.cpu().numpy()
    
    # Scale to 0-255 if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
        
    return Image.fromarray(img_array)

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]



def detect_objects(img, prompt):
    
    bounding_box_system_instructions = """
        You are an expert at analyzing images to identify and locate objects.
        Return bounding boxes as a JSON array. Each object in the array should have a "label" (string) and "box_2d" (array of 4 numbers).
        The "box_2d" coordinates must be [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale.
        Never return Python code fencing (```python ... ```) or general markdown fencing (``` ... ```) around the JSON. Only output the raw JSON array.
        If an object is present multiple times, name them uniquely (e.g., "red car", "red car 2") """
        
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]

          
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, img],
        config = types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(
              thinking_budget=0
            )
        )
    )

    return response.text


def bbox_2d_gemini(img, prompt):
    if not isinstance(img, Image.Image):
        img = tensor_to_pil(img)
    bounding_boxes = detect_objects(img, prompt)
    return bounding_boxes


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.
    
    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """
    
    # Load the image
    img = im
    width, height = img.size
    # print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)
    
    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors
    
    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)
    # print(bounding_boxes)
            
    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
      # Select a color from the list
      color = colors[i % len(colors)]
      
      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
      abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
      abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
      abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
      
      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
      
      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
      
      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=10
      )
      
      try:
          font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
      except ValueError:
          # Fallback to a generic sans-serif if DejaVu Sans isn't found
          font_path = fm.findfont(fm.FontProperties(family='sans-serif'))
      
      from PIL import ImageFont
      font = ImageFont.truetype(font_path, 30)
      
      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)
    
    # Display the image
    return img
    
def plot_2d_bbox(im, bounding_boxes):
    if not isinstance(im, Image.Image):
        im = tensor_to_pil(im)
    img = plot_bounding_boxes(im, bounding_boxes)
    return img
    
# img = "C:/Users/user/Downloads/test_image.png"
# img = Image.open(BytesIO(open(img, "rb").read()))
# prompt = "Detect the 2d bounding boxes of cars and carts" 
# bbox_2d_gemini(img, prompt)
# print(bbox, label)
# plot_2d_bbox(img, bbox, label)


