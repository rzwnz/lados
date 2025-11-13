// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:sample_app/main.dart';

void main() {
  testWidgets('App loads correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const LadosClassifierApp());

    // Verify that the app title is present
    expect(find.text('LADOS Classifier'), findsOneWidget);
    
    // Verify that the welcome message is present
    expect(find.text('Upload an image to classify using the LADOS model'), findsOneWidget);
  });
}
