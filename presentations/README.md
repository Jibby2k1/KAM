# KAM Phase VI presentations

The current professor-facing deck is:

- `KAM_Phase6_Professor_Presentation_v2.pptx`
- `KAM_Phase6_Professor_Presentation_v2.pdf`
- `KAM_Phase6_Professor_Deck_Outline_v2.md`

The deck pairs every explanatory visual with its associated routing,
retrieval, optimization, lifecycle, or statistical equation.

To regenerate the source artwork and PowerPoint:

```bash
cd presentations
npm install
npm run build:svg
for f in kam_phase6_professor_assets_v2/slide-*.svg; do
  convert -background none -density 120 "$f" -resize 1600x900 "${f%.svg}.png"
done
npm run build:pptx
