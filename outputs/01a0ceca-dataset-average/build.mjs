import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const dir = 'C:/work/paper_repo/3D_irregular_packing_stream/outputs/01a0ceca-dataset-average';
const csv = await fs.readFile('C:/work/paper_repo/3D_irregular_packing_stream/hpc result/summary_by_dataset.csv','utf8');
const imported = await Workbook.fromCSV(csv,{sheetName:'Imported'});
const data = imported.worksheets.getItemAt(0).getUsedRange().values;
const headers=data[0];
const rows=data.slice(1).filter(r=>r[0]).map(r=>r.map((v,i)=>i>=6 && ![9,10,11].includes(i) ? (v===''||v==null?null:Number(v)):v));
const metrics=headers.map((h,i)=>[h,i]).filter(([h])=>h.endsWith('_mean'));
const strategies=[...new Set(rows.map(r=>r[3]))];
const selections=[...new Set(rows.map(r=>r[4]))];
const shapes=[...new Set(rows.map(r=>r[1]))];
const datasets=[...new Set(rows.map(r=>r[0]))];
if(rows.length!==360 || datasets.length!==15) throw Error('Unexpected coverage');
const keys=new Set(rows.map(r=>[r[0],r[1],r[3],r[4],r[5]].join('|')));
if(keys.size!==rows.length || rows.some(r=>r[8]!==5 || metrics.some(([,i])=>!Number.isFinite(r[i])))) throw Error('Duplicate/missing data');
const wb=Workbook.create();
const out=wb.worksheets.add('Method averages');
const src=wb.worksheets.add('Source data');
src.getRange('A1').values=[['Source: hpc result/summary_by_dataset.csv']];
src.getRange('A3:AJ363').values=[headers,...rows];
out.getRange('A2').values=[['跨数据集的方法平均表现']];
out.getRange('A3').values=[['等权平均：每个 dataset 权重相同；Overall 同时等权合并 cube 和 cylinder。']];
out.getRange('A4').values=[['方法 = nesting_strategy × selection_range；constraint_mode 均为 geometry_only。']];
out.getRange('A5').values=[['各指标取源表 *_mean 的算术平均；不对源表标准差取平均。耗时单位为秒。']];
const col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const controls=[];
function section(start,title,shapeList){
 out.getRange(`A${start}`).values=[[title]];
 out.getRange(`A${start+1}:N${start+1}`).values=[['bin_shape','nesting_strategy','selection_range','datasets','source_rows','successful_runs',...metrics.map(([h])=>h)]];
 let r=start+2;
 for(const shape of shapeList)for(const strategy of strategies)for(const selection of selections){
  const group=rows.filter(x=>x[3]===strategy && x[4]===selection && (shape==='Overall'||x[1]===shape));
  out.getRange(`A${r}:D${r}`).values=[[shape,strategy,selection,new Set(group.map(x=>x[0])).size]];
  const criteria=`'Source data'!$D$4:$D$363,$B${r},'Source data'!$E$4:$E$363,$C${r}`+(shape==='Overall'?'':`,'Source data'!$B$4:$B$363,$A${r}`);
  out.getRange(`E${r}:N${r}`).formulas=[[`=COUNTIFS(${criteria})`,`=SUMIFS('Source data'!$I$4:$I$363,${criteria})`,...metrics.map(([,i])=>`=AVERAGEIFS('Source data'!$${col(i)}$4:$${col(i)}$363,${criteria})`)]];
  controls.push({r,expected:[group.length,group.reduce((s,x)=>s+x[8],0),...metrics.map(([,i])=>group.reduce((s,x)=>s+x[i],0)/group.length)]});r++;
 }
 out.getRange(`A${start+1}:N${start+1}`).format={fill:'#243E59',font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:34,horizontalAlignment:'center'};
 out.getRange(`D${start+2}:F${r-1}`).setNumberFormat('0');
 out.getRange(`G${start+2}:N${r-1}`).setNumberFormat('0.0000');
 for(let k=start+2;k<r;k++)if((k-start)%2===0)out.getRange(`A${k}:N${k}`).format.fill='#F0F4F8';
}
section(7,'全部箱体形状', ['Overall']);
section(23,'按箱体形状分别汇总',shapes);
for(const sheet of [out,src]){
 sheet.showGridLines=false;
 sheet.getUsedRange().format.font.name='Arial';sheet.getUsedRange().format.font.size=10;
 sheet.getUsedRange().format.verticalAlignment='center';
 sheet.getUsedRange().format.rowHeight=21;
}
out.getRange('A2').format.font={size:16,bold:true};
out.getRange('A7').format.font.bold=true;out.getRange('A23').format.font.bold=true;
out.getRange('A8:N8').format.rowHeight=36;out.getRange('A24:N24').format.rowHeight=36;
out.getRange('A1:A48').format.columnWidth=14;
out.getRange('B1:B48').format.columnWidth=29;
out.getRange('C1:C48').format.columnWidth=18;
out.getRange('D1:F48').format.columnWidth=13;
out.getRange('F1:F48').format.columnWidth=18;
out.getRange('G1:N48').format.columnWidth=20;
src.getRange('A3:AJ3').format={fill:'#243E59',font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42};
src.getRange('A3:AJ363').format.columnWidth=22;
src.getRange('D3:D363').format.columnWidth=29;
src.getRange('M4:AJ363').setNumberFormat('0.0000');
src.freezePanes.freezeRows(3);out.freezePanes.freezeRows(8);
wb.recalculate();
for(const {r,expected} of controls){const actual=out.getRange(`E${r}:N${r}`).values[0];actual.forEach((v,i)=>{if(typeof v!=='number'||Math.abs(v-expected[i])>1e-8*Math.max(1,Math.abs(expected[i])))throw Error(`Mismatch ${r}/${i}: ${v} ${expected[i]}`);});}
console.log('Verified all 360 calculated summary cells against independent arithmetic.');
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
for(const [sheet,range,name]of [['Method averages','A7:N20','summary'],['Source data','A3:J9','source']]){
 const img=await wb.render({sheetName:sheet,range,scale:1.5,format:'png'});await fs.writeFile(`${dir}/${name}.png`,new Uint8Array(await img.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(`${dir}/method_average_across_datasets.xlsx`);
console.log('Saved method_average_across_datasets.xlsx');
