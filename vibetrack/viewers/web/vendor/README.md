# Vendored third-party JS

These files are downloaded once and committed so the web UI works offline /
air-gapped. Do not edit by hand — re-run `fetch_vendor.sh` to refresh.

| File             | Source                                                                                                 | Version | License |
|------------------|--------------------------------------------------------------------------------------------------------|---------|---------|
| `three.min.js`   | https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js                                          | r128    | MIT     |
| `OrbitControls.js` | https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js                       | r128    | MIT     |

`OrbitControls.js` is the non-module ES5 build that attaches
`THREE.OrbitControls` to the global `THREE` namespace. It must be loaded
*after* `three.min.js`.
