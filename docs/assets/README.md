# HyperMEM visual assets

Brand assets for the HyperMEM repo, in the project's visual identity
(amber `#C26A16` / `#E08A3C` on a cool slate ground, the "memory thread"
motif, serif display + mono labels). The `.svg` sources are the canonical
files — edit those, then re-render the `.png` with `sharp`:

```bash
cd docs/assets
node -e "require('sharp')('banner.svg',{density:144}).png().toFile('banner.png')"
```

| File | Size | Purpose |
|------|------|---------|
| `banner.png` | 1280×640 | Hero image at the top of the README. |
| `og-image.png` | 1280×640 | GitHub **social preview** (repo → Settings → Social preview). Set to this file so the repo card shows the brand when shared. |
| `footer.png` | 1280×180 | Closing strip at the bottom of the README. |

All three share one identity: the thread (a vertical conversation spine with
lit knots at the pipeline steps) and the headline "AI memory that never
forgets."
