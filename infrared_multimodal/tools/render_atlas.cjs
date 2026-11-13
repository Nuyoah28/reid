// Use the bundled Playwright runtime; do not install anything into mmi-unet.
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');
const { chromium } = require(process.env.REID_PLAYWRIGHT_MODULE || 'C:/Users/zbw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/playwright-core@1.61.0/node_modules/playwright-core');
const root = path.resolve(__dirname, '..');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1564, height: 1100 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(pathToFileURL(path.join(root, 'research/architecture/network.html')).href);
  await page.evaluate(() => document.fonts.ready);
  const overflow = await page.locator('svg g[data-box]').evaluateAll(groups => groups.flatMap(g => {
    const [x,y,w,h] = g.dataset.box.split(',').map(Number);
    return [...g.querySelectorAll('text')].flatMap(t => {
      const b = t.getBBox();
      return b.x+b.width > x+w-8 || b.y+b.height > y+h-3 ? [{text:t.textContent, excess:b.x+b.width-(x+w-8)}] : [];
    });
  }));
  const flowOverflow = await page.locator('#every-block .blockflow, #every-block .blockflow-mobile').evaluateAll(svgs => svgs.flatMap(svg => {
    const [, , width, height] = svg.getAttribute('viewBox').split(/\s+/).map(Number);
    return [...svg.querySelectorAll('text')].flatMap(t => {
      const b=t.getBBox();
      return b.x < 0 || b.y < 0 || b.x+b.width > width || b.y+b.height > height ? [{diagram:svg.getAttribute('aria-label'),text:t.textContent}] : [];
    });
  }));
  const hideNavForFigureCapture = await page.addStyleTag({content:'nav{visibility:hidden!important}'});
  for (const name of ['baseline','bcrt','bottleneck','nonlocal','transport']) {
    await page.locator(`#${name} svg`).screenshot({ path: path.join(root,`research/architecture/${name}.png`) });
  }
  await hideNavForFigureCapture.evaluate(e => e.remove());
  const blockFlows = await page.locator('#every-block .blockflow').count();
  if (blockFlows !== 16) throw new Error(`Expected 16 per-block drawings, got ${blockFlows}`);
  if (await page.locator('#every-block .blockflow-mobile').count() !== 16) throw new Error('Missing narrow-screen diagrams');
  await page.locator('#every-block > .blocks').first().screenshot({path:path.join(root,'research/architecture/bottleneck_stage1.png')});
  await page.screenshot({path:path.join(root,'tmp/atlas_desktop.png')});
  await page.locator('#alpha').fill('0.1');
  await page.locator('#alpha').dispatchEvent('input');
  const toy = await page.locator('#toyResult').textContent();
  if (toy !== '0.200 / 0.300') throw new Error(`Unexpected geometry example: ${toy}`);
  await page.locator('#trace summary').click();
  await page.locator('#traceSearch').fill('Conv2d');
  const filteredRows = await page.locator('#traceTable tbody tr:visible').count();
  if (filteredRows < 50) throw new Error('Missing convolution trace rows');
  await page.setViewportSize({width:390,height:844});
  const mobileCards = page.locator('#every-block > .blocks').first();
  const hideNavForMobileCapture = await page.addStyleTag({content:'nav{visibility:hidden!important}'});
  await mobileCards.screenshot({path:path.join(root,'tmp/atlas_mobile_blocks.png')});
  await hideNavForMobileCapture.evaluate(e => e.remove());
  const mobileFlow = await page.locator('#every-block .blockflow-mobile').first().evaluate(e => ({width:e.clientWidth,scrollWidth:e.scrollWidth,container:e.parentElement.clientWidth}));
  await page.evaluate(() => scrollTo(0,0));
  await page.screenshot({path:path.join(root,'tmp/atlas_mobile.png')});
  const bodyOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  const report = {errors, boxTextOverflow:overflow, perBlockTextOverflow:flowOverflow,perBlockDrawings:blockFlows,mobileBlockDrawings:16,mobileBlockFlow:mobileFlow,filteredConvCalls:filteredRows,mobileBodyOverflow:bodyOverflow,geometryControl:toy};
  fs.writeFileSync(path.join(root,'research/architecture/render_check.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  await browser.close();
  if(errors.length || overflow.length || flowOverflow.length || bodyOverflow) process.exitCode=1;
})().catch(e => {console.error(e);process.exit(1)});
