// Build with the supplied @oai/artifact-tool runtime. All evidence charts are editable.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { FileBlob, Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = process.cwd();
const SKILL = process.env.PRESENTATION_SKILL_DIR;
if (!SKILL) throw new Error('Set PRESENTATION_SKILL_DIR to the installed presentation skill.');
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(
  path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
// Ten significant digits keep editable workbooks within Excel's numeric precision.
// Full-precision analytical results stay unchanged in results/*.csv.
const D = JSON.parse(await fs.readFile(path.join(ROOT, 'results/presentation_data.json'), 'utf8'),
  (_,v)=>typeof v==='number'?Number(v.toPrecision(10)):v);
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { ink: '#173447', muted: '#4e6472', blue: '#24668d', red: '#c44945', paper: '#fafbf9', line: '#dde3e7' };
const FONT = 'Arial';
const pct = (v, n=2) => `${(100*v).toFixed(n)}%`;
const num = n => Number(n).toLocaleString('en-US');

function text(s, value, x, y, w, h, size=28, color=C.ink, bold=false) {
  const sh = s.shapes.add({ geometry:'textbox', position:{left:x, top:y, width:w, height:h},
    fill:'none', line:{fill:'none', width:0} });
  sh.text = value;
  sh.text.style = { typeface:FONT, fontSize:size, color, bold, verticalAlignment:'middle' };
  return sh;
}
function slide(title, notes='') {
  const s = p.slides.add(); s.background.fill=C.paper;
  text(s, title, 64, 38, 1140, 100, 44, C.ink, true);
  text(s, String(p.slides.items.length).padStart(2,'0'), 1160, 668, 50, 28, 18, C.muted);
  s.speakerNotes.text=notes;
  return s;
}
function paragraph(s, value, y, size=28, color=C.ink) { return text(s,value,72,y,1120,88,size,color); }
function table(s, values, x=72, y=170, w=1120, h=300, widths) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,
    values, ...(widths?{columnWidths:widths}:{})});
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
    textStyle:{typeface:FONT,fontSize:25,color:C.ink}, margins:{left:14,right:12,top:12,bottom:12}, anchor:'center' });
  for(let col=0; col<values[0].length; col++) {
    const cell=t.getCell(0,col); cell.fill=C.blue;
    cell.text.style={typeface:FONT,fontSize:25,bold:true,color:'#ffffff'};
  }
  return t;
}
function chart(s,type,config) {
  const ch=s.charts.add(type,{
    position:{left:66,top:160,width:780,height:440},
    chartFill:C.paper,plotAreaFill:C.paper,
    hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:22,fill:C.muted}},
    titleTextStyle:{fontSize:25,fill:C.ink,bold:true},
    xAxis:{textStyle:{fontSize:21,fill:C.muted},line:{fill:C.line,width:1}},
    yAxis:{textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.line,width:1}},
    ...config,
    ...(type==='line'?{lineOptions:{smooth:false},series:config.series.map(v=>({...v,smooth:false}))}:{}),
  });
  applyPresentationChartFont(ch,{fontFamily:FONT}); return ch;
}

let s=slide('Card fraud triage', 'Team T011, Section A. H Maheshwari 2401010174, Vaibhav 2401010486, Mayank 2401010267, Keshav 2401010222, Mitul Bhatia 2401010279. Explain that this deck presents Phase 1 only. Source: supplied allocation email and course PDF page 2.');
text(s,'Gaussian mixture density scoring',72,214,1100,90,52,C.blue,true);
text(s,'Phase 1: Machine Learning',72,334,1100,55,34);
text(s,'Team T011, Section A, Project 26',72,467,1100,55,29);
text(s,'Newton School of Technology, Rishihood University',72,535,1100,55,26,C.muted);

s=slide('The review task','Say: We learn the normal patterns of known legitimate transactions. We rank later transactions by unusualness and send a limited number for manual review. A high anomaly score does not prove fraud. Course PDF page 2 restricts this evaluation to ML. No autoencoder or hybrid model is implemented.');
paragraph(s,'Learn normal activity from earlier legitimate transactions.',180,33);
paragraph(s,'Give less typical transactions a larger anomaly score.',285,33);
paragraph(s,'Measure fraud found and legitimate transactions sent for review.',390,33);
paragraph(s,'Phase 1 model: GMM. Autoencoder comparison belongs to Phase 2.',535,27,C.muted);

s=slide('Literature behind our choices','Dempster, Laird and Rubin (1977), JRSS B 39(1), pp 1-22, DOI https://doi.org/10.1111/j.2517-6161.1977.tb01600.x. Saito and Rehmsmeier (2015), PLOS ONE 10(3), e0118432, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432. scikit-learn mixture guide https://scikit-learn.org/stable/modules/mixture.html and selection example https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html. These sources explain methods and evaluation, not our results.');
table(s,[['Source','Idea','Use in our project'],['Dempster et al., 1977','EM estimates hidden memberships','Fit the Gaussian components'],['Saito & Rehmsmeier, 2015','PR reveals rare-class alert burden','Lead with PR and review metrics'],['scikit-learn mixture guide','Covariance and information criteria','Diagonal GMM, AIC and BIC']],72,176,1120,330,[330,395,395]);
paragraph(s,'Our contribution is a clear, reproducible baseline experiment.',540,28,C.muted);

s=slide('Dataset quality','Sources: ULB/Worldline dataset https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and documented CSV mirror https://www.tensorflow.org/tutorials/structured_data/imbalanced_data. Actual file measurements are in results/data_quality.json. We found 0 missing and infinite cells. Remove exact full-row repeats and keep the first. There is no transaction ID, so this is an assumption. Do not call all repeated rows verified duplicate purchases. V1-V28 are anonymous provider PCA features.');
table(s,[['Measured quantity','Original CSV','After exact-repeat removal'],['Transactions',num(D.quality.raw_rows),num(D.quality.clean_rows)],['Fraud labels',num(D.quality.raw_frauds),num(D.quality.clean_frauds)],['Exact repeats removed','1,081','19 of these carry fraud labels']],72,174,1120,300,[370,310,440]);
paragraph(s,'31 columns. No missing or infinite cells. Source checksum recorded.',515,27);
paragraph(s,'V1-V28 are anonymous PCA inputs. Business meanings are unavailable.',594,25,C.muted);

s=slide('Chronological evaluation','Source: results/splits.csv. Split around 60/20/20 of cleaned rows. Ties in Time stay within one period. Train Time <120393 seconds, validation 120393-145233, test 145234-172792. Scaler and GMM use only 169893 legitimate training rows. Thresholds use validation scores. Test labels do not choose features, K or thresholds. Base-rate reporting does not fit the model.');
table(s,[['Period','Rows','Legitimate','Fraud'],...D.splits.map(x=>[x.split,num(x.rows),num(x.legitimate),String(x.fraud)])],72,174,1120,298,[320,270,270,260]);
paragraph(s,'Fit on 169,893 legitimate training rows. Freeze choices before test.',516,30,C.blue);
paragraph(s,'Equal timestamps stay together. No future rows fit preprocessing.',597,26,C.muted);

s=slide('Training amount distribution','Source: training rows after the documented split and cleaning, generated by scripts/build_presentation_data.py. Native chart shows fraction within each class per 0.5-wide log-amount bin from 0 to 12. Density shapes are normalized separately, so curve heights do not show class counts. Amount uses EUR. log(1+Amount) accepts zero and reduces the long tail. All model inputs remain untrimmed.');
chart(s,'line',{title:'Share within each class by log-amount bin',categories:D.amount.bins.map(x=>x.toFixed(2)),
 series:[{name:'Legitimate',values:D.amount.legitimate,line:{fill:C.blue,width:3}},
         {name:'Fraud',values:D.amount.fraud,line:{fill:C.red,width:3}}],
 xAxis:{title:'log(1 + Amount), bin midpoint',textStyle:{fontSize:17,fill:C.muted}},
 yAxis:{title:'Share within class',numberFormatCode:'0%',min:0,textStyle:{fontSize:21},majorGridlines:{fill:C.line,width:1}}});
text(s,'Amount spans a wide range.\n\nThe log transform compresses the tail and handles zero.',890,217,310,290,29);

s=slide('The 29 model inputs','Features: V1-V28 plus log(1+Amount). No added PCA. StandardScaler fits only known legitimate training rows. V1-V28 were already processed by the provider, whose original PCA fit we cannot audit. Filtering legitimate history uses labels, although GMM fitting itself is unsupervised. Explain this nuance if asked. Source: src/project.py make_features and prepare_features.');
paragraph(s,'V1-V28 plus LogAmount = log(1 + Amount)',178,36,C.blue);
paragraph(s,'Time orders the split. Class filters legitimate history and evaluates results.',287,29);
paragraph(s,'z = (value - legitimate training mean) / legitimate training standard deviation',400,29);
paragraph(s,'Validation and test use these saved training values.',520,31);
paragraph(s,'Class and Time never enter the model feature matrix.',597,27,C.muted);

s=slide('How the GMM learns','Source: model equations in report/phase1_report.tex, implementation in src/project.py, and https://scikit-learn.org/stable/modules/mixture.html. A weighted sum of Gaussian densities allows multiple centers. E-step estimates responsibilities, M-step updates means, diagonal variances and weights. Three starts and variance regularization improve practical fitting but do not guarantee a global optimum. The mixture can express overall dependence despite diagonal within-component covariance.');
paragraph(s,'p(x) = sum of [component weight x Gaussian density]',170,35,C.blue);
paragraph(s,'E-step: estimate soft membership in each Gaussian.',278,31);
paragraph(s,'M-step: update weights, centers and diagonal variances.',376,31);
paragraph(s,'Three starts. Seed 42. Up to 200 iterations. Small variance regularization.',485,28);
paragraph(s,'Anomaly score = -log p(x). Larger means less typical of legitimate activity.',584,27,C.muted);

s=slide('Component selection','Source: results/gmm_selection.csv. AIC=2q-2L and BIC=q log N-2L. All candidates fit the same 169893 legitimate training rows, with diagonal covariance. Select minimum training BIC among converged candidates. Every candidate converged. K=8 has 471 free parameters for d=29. Both criteria still decrease at the upper grid boundary. Better legitimate density fit does not guarantee better fraud ranking.');
chart(s,'line',{title:'Training criteria, lower is better',categories:D.selection.map(x=>String(x.components)),
 series:[{name:'AIC / 1 million',values:D.selection.map(x=>Number((x.aic/1e6).toPrecision(10))),line:{fill:C.blue,width:3}},
         {name:'BIC / 1 million',values:D.selection.map(x=>Number((x.bic/1e6).toPrecision(10))),line:{fill:C.red,width:3}}],
 xAxis:{title:'Number of Gaussian components',textStyle:{fontSize:22}},
 yAxis:{title:'Criterion / 1 million',textStyle:{fontSize:21},majorGridlines:{fill:C.line,width:1}}});
text(s,'K = 8 wins in our grid.\n\nAll candidates converge.\n\nEight is the largest value tested.',890,210,310,350,29);

s=slide('Final test ranking','Sources: results/ranking_metrics.csv and test_scores.csv. The native chart connects retained PR points from sklearn precision_recall_curve(drop_intermediate=True). Values are actual held-out mixed-test results. AP is step-weighted, trapezoidal PR AUC uses linear integration, and neither is accuracy. The reference one-Gaussian model uses the same legitimate training rows and scaler. Test prevalence is 74/56746=0.1304%.');
chart(s,'scatter',{title:'Selected GMM precision-recall curve',scatterOptions:{style:'line'},
 series:[{name:'GMM',xValues:D.pr.recall,values:D.pr.precision,line:{fill:C.blue,width:2.5},marker:{symbol:'none'}},
 {name:'Test prevalence',xValues:[0,1],values:[D.ranking[0].fraud_rate,D.ranking[0].fraud_rate],line:{fill:C.red,width:1.5},marker:{symbol:'none'}}],
 xAxis:{title:'Recall',min:0,max:1,majorUnit:0.2,numberFormatCode:'0%',textStyle:{fontSize:21}},
 yAxis:{title:'Precision',min:0,max:1,majorUnit:0.2,numberFormatCode:'0%',textStyle:{fontSize:21},majorGridlines:{fill:C.line,width:1}}});
text(s,'Average precision\nGMM: 0.1532\nOne Gaussian: 0.0540\n\nPR AUC: 0.1496\nTest fraud: 0.1304%',890,202,310,360,28);

s=slide('Validation policies on the test period','Source: results/test_operating_points.csv and validation_thresholds.csv. Budgets refer to empirical validation FPR, never guaranteed future rates. Thresholds come from legitimate validation scores, with strict score > threshold and conservative ties. Default 0.5% was fixed as an illustrative classroom policy before test outcomes. Precision at the three thresholds is 13.33%,17.70%,12.01%. Do not suggest that 0.5% was selected by maximizing test precision.');
table(s,[['Validation budget','Fraud found / 74','False alerts','Test FPR'],...D.operating.map(x=>[pct(x.fpr_budget,1),String(x.tp),String(x.fp),pct(x.fpr,4)])],72,170,1120,300,[325,295,240,260]);
paragraph(s,'Default 0.5% policy: threshold 75.8942, with score > threshold.',513,29,C.blue);
paragraph(s,'A validation budget may give a different false-positive rate later.',596,27,C.muted);

s=slide('The default review workload','Source: results/test_operating_points.csv, default fpr_budget=.005. TN56500, FP172, FN37, TP37, alerts209. Recall=37/(37+37)=50%, precision=37/(37+172)=17.70%, FPR=172/(172+56500)=0.3035%. The model misses 37 frauds and most alerts are legitimate. Say these limits explicitly.');
chart(s,'bar',{title:'Outcomes among 209 test alerts',categories:['Fraud found','Legitimate alerted'],
 series:[{name:'Transactions',values:[37,172],fill:C.blue}],hasLegend:false,
 barOptions:{direction:'column'},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:25}},
 xAxis:{textStyle:{fontSize:23}},yAxis:{title:'Transactions',min:0,max:200,textStyle:{fontSize:22},majorGridlines:{fill:C.line,width:1}}});
text(s,'Recall: 37 / 74 = 50%\n\nPrecision: 37 / 209 = 17.70%\n\nMissed frauds: 37',876,215,335,330,28);

s=slide('Fixed batch review capacity','Source: results/test_review_capacity.csv. Review the highest scoring K rows of this entire test batch. Break score ties by existing chronological order. At K=100, 29/100 fraud, 29/74 recall=39.19%. At K50 precision=.16, K100=.29, K200=.185, K500=.106. Precision need not decrease monotonically because the next block can have different fraud prevalence. Capacity is an alternative to a fixed score threshold, not a daily live queue claim.');
chart(s,'line',{title:'Precision at a fixed test-batch capacity',categories:D.capacity.map(x=>String(x.capacity)),
 series:[{name:'Precision@K',values:D.capacity.map(x=>x.precision_at_k),line:{fill:C.blue,width:3},marker:{symbol:'circle',size:8,fill:C.blue}}],hasLegend:false,
 xAxis:{title:'Transactions reviewed (K)',textStyle:{fontSize:22}},
 yAxis:{title:'Precision@K',numberFormatCode:'0%',min:0,max:.35,textStyle:{fontSize:22},majorGridlines:{fill:C.line,width:1}}});
text(s,'Top 100 reviews\n\n29 frauds found\n\n29% precision\n39.19% recall',890,212,310,340,30);

s=slide('Limits and Phase 1 conclusion','Our measured GMM improves PR ranking over one Gaussian in this run. It is an educational baseline, not a production claim. Limits: only 74 test frauds (1 fraud changes recall by about1.35 percentage points), short historical dataset, anonymous PCA inputs, no audit of provider PCA, duplicate-removal assumption, diagonal covariance, largest-K grid boundary, no repeated-window uncertainty estimate. Phase 2 only: train an autoencoder on the same legitimate training rows and compare under the same split, validation policy and metrics. Source: measured tables and report limitations. End with a short notebook demo and individual questions.');
paragraph(s,'The GMM improves this ranking over one Gaussian.',171,34,C.blue);
paragraph(s,'At the default policy, it catches half the test fraud and creates 172 false alerts.',275,29);
paragraph(s,'Only 74 test frauds. Anonymous inputs. Short history. Restrictive density assumptions.',390,29);
paragraph(s,'Phase 2: compare an autoencoder under the same data and evaluation protocol.',522,29);
paragraph(s,'Every member should explain the split, score, threshold and exact metric counts.',607,25,C.muted);

const build=path.join(ROOT,'.build/slides');
await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
// Native literal evidence charts intentionally get embedded data workbooks.
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:path.join(ROOT,'output/phase1_presentation_final.pptx'),
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
    ...[3,4,5,11].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:14,
  requiredNativeTableOwnerSlides:[3,4,5,11],requiredNativeChartOwnerSlides:[6,9,10,12,13],
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'validation_final.json'),
});
console.log(JSON.stringify(result));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(path.join(ROOT,'output/phase1_presentation_final.pptx')));
for (let i=0;i<finalDeck.slides.items.length;i++) {
  const img=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await img.arrayBuffer()));
}
console.log('Exported and rendered 14 editable slides.');
