# LADOS Classifier - Flutter Web App

Flutter web application for the LADOS oil spill classification system. Provides an intuitive interface for uploading images, viewing predictions, and monitoring system metrics.

## Features

- ✅ Material Design 3 UI
- ✅ Image upload and preview
- ✅ Single image prediction
- ✅ Batch prediction support
- ✅ Interactive bar charts (fl_chart)
- ✅ Real-time metrics display
- ✅ Training metrics visualization
- ✅ Responsive design

## Prerequisites

- Flutter SDK (3.7.2 or higher)
- FastAPI backend running on http://localhost:8000
- Chrome browser (for web development)

## Setup

1. **Install Flutter dependencies:**
   ```bash
   cd flutter/sample_app
   flutter pub get
   ```

2. **Ensure Flutter is in your PATH:**
   ```bash
   source ~/.zshrc  # or ~/.bashrc depending on your shell
   ```

## Running the App

### Development Mode

```bash
flutter run -d chrome --web-port 8080
```

The app will open in Chrome at http://localhost:8080

### Production Build

```bash
# Build for production
flutter build web --release

# The built files will be in build/web/
# You can serve them with any static file server
```

## Usage

1. **Select Image**: Click "Select Image" button to choose an image from your device
2. **View Preview**: Selected image will be displayed in the preview area
3. **Predict**: Click "Predict" button to send the image to the API
4. **View Results**: 
   - Top prediction with confidence score
   - Interactive bar chart showing top 5 predictions
   - Detailed list with progress bars
5. **Batch Predict**: Use "Batch Predict" to select multiple images (feature in development)

## API Integration

The app connects to the FastAPI backend at `http://localhost:8000` and uses the following endpoints:

- `GET /health` - Health check (not used in UI, but can be checked)
- `GET /metrics` - Training and inference metrics
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction (for future implementation)

### Configuration

To change the API URL, edit the `apiUrl` variable in `lib/main.dart`:

```dart
final String apiUrl = 'http://localhost:8000';
```

## Project Structure

```
lib/
└── main.dart          # Main application code
web/
├── index.html        # Web entry point
└── manifest.json     # Web app manifest
```

## Troubleshooting

### White Screen

If you see a white screen:

1. **Check browser console** (F12) for JavaScript errors
2. **Verify FastAPI is running**: `curl http://localhost:8000/health`
3. **Check CORS settings** in FastAPI (should allow all origins in dev)
4. **Rebuild the app**:
   ```bash
   flutter clean
   flutter pub get
   flutter build web
   ```

### Image Picker Not Working

- On web, the image picker uses the browser's file input
- Some browsers may have restrictions on file access
- Try a different browser if issues persist

### API Connection Errors

- Ensure FastAPI server is running: `docker compose ps api`
- Check API logs: `docker compose logs api`
- Verify the API URL in `lib/main.dart` matches your server

### Build Errors

- Ensure all dependencies are installed: `flutter pub get`
- Check Flutter version: `flutter --version` (should be ≥3.7.2)
- Try cleaning and rebuilding: `flutter clean && flutter pub get`

## Dependencies

- `http` - HTTP client for API calls
- `image_picker` - Image selection
- `file_picker` - File selection (for batch)
- `fl_chart` - Chart visualization

## Development

### Hot Reload

While running in development mode, you can use hot reload:
- Press `r` in the terminal to hot reload
- Press `R` to hot restart

### Debugging

- Use browser DevTools (F12) for debugging
- Check Flutter DevTools for widget inspection
- View network requests in browser Network tab

## Known Issues

- `file_picker` package warnings (cosmetic, doesn't affect functionality)
- Batch prediction feature is not fully implemented yet
- Image picker on web has some browser-specific limitations

## Related Documentation

- Main project README: [../../README.md](../../README.md)
- Service health checks: [../../SERVICE_HEALTH_CHECK.md](../../SERVICE_HEALTH_CHECK.md)
- Dataset information: [../../data/README.md](../../data/README.md)
