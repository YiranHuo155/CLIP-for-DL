<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="500" fill="#f9f9f9" rx="10" ry="10"/>
  
  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">CLIP-based Model with Lightweight Classification Adapter</text>

  <!-- CLIP Framework Box (Expanded to include Fusion) -->
  <rect x="80" y="80" width="520" height="340" fill="none" stroke="#8e44ad" stroke-width="2" stroke-dasharray="5,5" rx="10" ry="10"/>
  <text x="340" y="105" text-anchor="middle" font-family="Arial" font-size="16" fill="#8e44ad" font-weight="bold">CLIP Framework</text>
  
  <!-- Image Encoder Section -->
  <rect x="100" y="130" width="200" height="80" fill="#d4e6f1" stroke="#3498db" stroke-width="2" rx="5" ry="5"/>
  <text x="200" y="170" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Image Encoder</text>
  <text x="200" y="195" text-anchor="middle" font-family="Arial" font-size="16">(ResNet)</text>
  
  <!-- Input Image -->
  <rect x="40" y="140" width="60" height="60" fill="#bdc3c7" stroke="#7f8c8d" stroke-width="2"/>
  <text x="70" y="170" text-anchor="middle" font-family="Arial" font-size="14">X-ray</text>
  <text x="70" y="190" text-anchor="middle" font-family="Arial" font-size="14">Image</text>
  
  <!-- Arrow from Image to Encoder -->
  <line x1="100" y1="170" x2="100" y2="170" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Text Encoder Section -->
  <rect x="100" y="280" width="200" height="80" fill="#d5f5e3" stroke="#2ecc71" stroke-width="2" rx="5" ry="5"/>
  <text x="200" y="320" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Text Encoder</text>
  <text x="200" y="345" text-anchor="middle" font-family="Arial" font-size="16">(Bio_ClinicalBERT)</text>
  
  <!-- Input Text -->
  <rect x="40" y="290" width="60" height="60" fill="#bdc3c7" stroke="#7f8c8d" stroke-width="2"/>
  <text x="70" y="320" text-anchor="middle" font-family="Arial" font-size="14">Radiology</text>
  <text x="70" y="340" text-anchor="middle" font-family="Arial" font-size="14">Report</text>
  
  <!-- Arrow from Text to Encoder -->
  <line x1="100" y1="320" x2="100" y2="320" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Paired Training Indicator -->
  <path d="M 70 200 L 70 290" fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="4,2"/>
  <text x="70" y="250" transform="rotate(90 70,250)" text-anchor="middle" font-family="Arial" font-size="12" fill="#e74c3c" font-weight="bold">Paired Training</text>
  
  <!-- Image Embedding -->
  <rect x="300" y="150" width="120" height="40" fill="#aed6f1" stroke="#3498db" stroke-width="2" rx="5" ry="5"/>
  <text x="360" y="175" text-anchor="middle" font-family="Arial" font-size="14">Image Embedding</text>
  
  <!-- Text Embedding -->
  <rect x="300" y="300" width="120" height="40" fill="#abebc6" stroke="#2ecc71" stroke-width="2" rx="5" ry="5"/>
  <text x="360" y="325" text-anchor="middle" font-family="Arial" font-size="14">Text Embedding</text>
  
  <!-- Arrows from Encoders to Embeddings -->
  <line x1="300" y1="170" x2="300" y2="170" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="300" y1="320" x2="300" y2="320" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Fusion Module -->
  <rect x="450" y="200" width="120" height="100" fill="#f9e79f" stroke="#f39c12" stroke-width="2" rx="5" ry="5"/>
  <text x="510" y="240" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Fusion</text>
  <text x="510" y="260" text-anchor="middle" font-family="Arial" font-size="14">Image-Text</text>
  <text x="510" y="280" text-anchor="middle" font-family="Arial" font-size="14">Representation</text>
  
  <!-- Arrows to Fusion -->
  <path d="M 420 170 L 450 170 L 450 230" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 420 320 L 450 320 L 450 270" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Lightweight Classification Adapter -->
  <rect x="630" y="200" width="150" height="100" fill="#fadbd8" stroke="#e74c3c" stroke-width="2" rx="5" ry="5"/>
  <text x="705" y="235" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">Lightweight</text>
  <text x="705" y="255" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">Classification</text>
  <text x="705" y="275" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">Adapter</text>
  <text x="705" y="295" text-anchor="middle" font-family="Arial" font-size="14">(FC Layer + Sigmoid)</text>
  
  <!-- Joint Representation Label -->
  <text x="595" y="240" text-anchor="middle" font-family="Arial" font-size="12" fill="#34495e" font-style="italic">Fused</text>
  <text x="595" y="255" text-anchor="middle" font-family="Arial" font-size="12" fill="#34495e" font-style="italic">Image-Text</text>
  <text x="595" y="270" text-anchor="middle" font-family="Arial" font-size="12" fill="#34495e" font-style="italic">Embedding</text>
  
  <!-- Arrow from Fusion to Adapter -->
  <line x1="570" y1="250" x2="630" y2="250" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Output -->
  <rect x="680" y="100" width="100" height="60" fill="#d7bde2" stroke="#8e44ad" stroke-width="2" rx="5" ry="5"/>
  <text x="730" y="130" text-anchor="middle" font-family="Arial" font-size="14">Multi-label</text>
  <text x="730" y="150" text-anchor="middle" font-family="Arial" font-size="14">Prediction [0,1]</text>
  
  <!-- Arrow from Adapter to Output -->
  <path d="M 705 200 L 705 160" fill="none" stroke="#34495e" stroke-width="2"/>
  <polygon points="705 160, 701 168, 709 168" fill="#34495e"/>
  
  <!-- Grad-CAM Visualization -->
  <rect x="500" y="400" width="200" height="50" fill="#d6eaf8" stroke="#3498db" stroke-width="2" rx="5" ry="5"/>
  <text x="600" y="430" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold">Grad-CAM Visualization</text>
  <text x="600" y="445" text-anchor="middle" font-family="Arial" font-size="10" font-style="italic">(Applied pre- and post-adapter)</text>
  
  <!-- Arrows for Grad-CAM (bi-directional to show comparison) -->
  <path d="M 705 300 L 705 360 L 600 360 L 600 400" fill="none" stroke="#34495e" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrowhead)"/>
  <path d="M 200 210 L 200 360 L 600 360 L 600 400" fill="none" stroke="#34495e" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrowhead)"/>
  
  <!-- Definitions for markers -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Legend -->
  <rect x="50" y="400" width="15" height="15" fill="#d4e6f1" stroke="#3498db" stroke-width="1"/>
  <text x="75" y="412" font-family="Arial" font-size="12" text-anchor="start">Image Processing</text>
  
  <rect x="50" y="425" width="15" height="15" fill="#d5f5e3" stroke="#2ecc71" stroke-width="1"/>
  <text x="75" y="437" font-family="Arial" font-size="12" text-anchor="start">Text Processing</text>
  
  <rect x="200" y="400" width="15" height="15" fill="#f9e79f" stroke="#f39c12" stroke-width="1"/>
  <text x="225" y="412" font-family="Arial" font-size="12" text-anchor="start">Fusion Module</text>
  
  <rect x="200" y="425" width="15" height="15" fill="#fadbd8" stroke="#e74c3c" stroke-width="1"/>
  <text x="225" y="437" font-family="Arial" font-size="12" text-anchor="start">Classification Adapter</text>
  
  <rect x="350" y="425" width="15" height="15" fill="none" stroke="#8e44ad" stroke-width="1" stroke-dasharray="3,1"/>
  <text x="375" y="437" font-family="Arial" font-size="12" text-anchor="start">CLIP Framework Boundary</text>
</svg>
