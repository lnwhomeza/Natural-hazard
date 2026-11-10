// ตัวช่วยสร้างเอกสาร Word ภาษาไทยด้วย docx-js
const fs = require('fs');
const path = require('path');
const sizeOf = (p) => {
  // อ่านขนาดภาพ PNG/JPEG โดยไม่ต้องพึ่งไลบรารีเพิ่ม
  const b = fs.readFileSync(p);
  if (b[0] === 0x89) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), type: 'png' };
  let i = 2;
  while (i < b.length) {
    const m = b[i + 1]; const len = b.readUInt16BE(i + 2);
    if (m >= 0xc0 && m <= 0xc3) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7), type: 'jpg' };
    i += 2 + len;
  }
  throw new Error('unknown image ' + p);
};
const d = require('docx');
const {
  Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, BorderStyle,
  ShadingType, PageBreak, TabStopType, VerticalAlign, HeadingLevel,
} = d;

const FONT = 'TH SarabunPSK';
const SZ = 32;          // 16 pt
const TEXT_W = 8306;    // A4 กว้าง 11906 − ซ้าย 2160 − ขวา 1440 (DXA)
const JUST = AlignmentType.THAI_DISTRIBUTE || AlignmentType.JUSTIFIED;

// แปลง **ตัวหนา** และ *ตัวเอียง* เป็น TextRun
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0; let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: SZ, ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), font: FONT, size: SZ, bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: FONT, size: SZ, italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: SZ, ...base }));
  return out;
}

const state = { toc: [], tables: [], figures: [], chapter: 0, nT: 0, nF: 0, nE: 0 };

function P(text, o = {}) {
  return new Paragraph({
    children: runs(text, o.run || {}),
    alignment: o.align || JUST,
    indent: o.noIndent ? undefined : { firstLine: o.firstLine ?? 720, left: o.left || 0 },
    spacing: { before: o.before || 0, after: o.after ?? 0 },
    keepNext: o.keepNext || false,
    pageBreakBefore: o.pageBreakBefore || false,
  });
}
const C = (text, o = {}) => P(text, { ...o, align: AlignmentType.CENTER, noIndent: true });
const blank = (n = 1) => Array.from({ length: n }, () => new Paragraph({ children: [new TextRun({ text: '', font: FONT, size: SZ })] }));

// หัวข้อบท (ขึ้นหน้าใหม่)
function chapter(no, title) {
  state.chapter = no; state.nT = 0; state.nF = 0; state.nE = 0;
  state.toc.push({ level: 0, text: `บทที่ ${no} ${title}`, key: `บทที่ ${no}` });
  return [
    new Paragraph({ children: [new TextRun({ text: `บทที่ ${no}`, font: FONT, size: 40, bold: true })],
      alignment: AlignmentType.CENTER, pageBreakBefore: true, heading: HeadingLevel.HEADING_1, spacing: { after: 0 } }),
    new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 40, bold: true })],
      alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
  ];
}
// หัวข้อหน้าส่วนหน้า/ท้าย (บทคัดย่อ บรรณานุกรม ฯลฯ)
function front(title, inToc = true, level = 0) {
  if (inToc) state.toc.push({ level, text: title, key: title });
  return new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 40, bold: true })],
    alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 240 }, heading: HeadingLevel.HEADING_1 });
}
function H2(num, text) {
  const t = `${num} ${text}`;
  state.toc.push({ level: 1, text: t, key: t });
  return new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 36, bold: true })],
    spacing: { before: 200, after: 80 }, keepNext: true, heading: HeadingLevel.HEADING_2 });
}
function H3(num, text) {
  const t = `${num} ${text}`;
  return new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: SZ, bold: true })],
    spacing: { before: 120, after: 0 }, keepNext: true, indent: { left: 360 }, heading: HeadingLevel.HEADING_3 });
}
// รายการข้อ 1) 2) ...
function items(list, o = {}) {
  return list.map((t, i) => new Paragraph({
    children: runs((o.bullet ? '• ' : `${o.start ? o.start + i : i + 1}) `) + t),
    alignment: JUST, indent: { left: (o.left || 720) + 360, hanging: 360 }, spacing: { after: 0 },
  }));
}
// สมการ (กึ่งกลาง + เลขสมการชิดขวา)
function eq(text) {
  state.nE += 1;
  const no = `(${state.chapter}.${state.nE})`;
  return new Paragraph({
    children: [new TextRun({ text: '\t' + text, font: 'Cambria Math', size: 26 }),
      new TextRun({ text: '\t' + no, font: FONT, size: SZ })],
    tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 80, after: 80 },
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  // keep: ให้ทั้งตารางขึ้นหน้าเดียวกัน (ตารางสั้น)
  const lines = String(text).split('\n');
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: o.head ? { fill: 'DCE6F1', type: ShadingType.CLEAR, color: 'auto' } : (o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: 'auto' } : undefined),
    margins: { top: 20, bottom: 20, left: 80, right: 80 },
    columnSpan: o.span,
    children: lines.map((l) => new Paragraph({
      children: runs(l, { size: o.size || 28, bold: !!o.head }),
      alignment: o.align || (o.head ? AlignmentType.CENTER : AlignmentType.LEFT),
      keepNext: !!o.keep,
    })),
  });
}
// ตาราง: caption ด้านบน
function table(caption, header, rows, widths, o = {}) {
  state.nT += 1;
  const label = `ตารางที่ ${state.chapter}.${state.nT}`;
  state.tables.push({ text: `${label} ${caption}`, key: label });
  const total = widths.reduce((a, b) => a + b, 0);
  const aligns = o.aligns || widths.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER));
  const tRows = [];
  const heads = Array.isArray(header[0]) ? header : [header];
  const keep = rows.length <= 8;
  heads.forEach((h) => tRows.push(new TableRow({ tableHeader: true, children: h.map((t, i) => cell(t, widths[i], { head: true, size: o.size, keep: true })) })));
  rows.forEach((r, j) => tRows.push(new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], { align: aligns[i], size: o.size, fill: o.fills && o.fills[i], keep: keep && j < rows.length - 1 })) })));
  const out = [
    new Paragraph({ children: [new TextRun({ text: label + ' ', font: FONT, size: SZ, bold: true }), ...runs(caption)],
      spacing: { before: 160, after: 60 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: tRows }),
  ];
  if (o.source) out.push(new Paragraph({ children: runs('ที่มา: ' + o.source, { size: 28 }), spacing: { before: 40, after: 120 } }));
  else out.push(new Paragraph({ children: [new TextRun({ text: '', size: 16 })], spacing: { after: 80 } }));
  return out;
}
// รูป: caption ด้านล่าง
function figure(file, caption, o = {}) {
  state.nF += 1;
  const label = `รูปที่ ${state.chapter}.${state.nF}`;
  state.figures.push({ text: `${label} ${caption}`, key: label });
  const s = sizeOf(file);
  const maxW = (o.width || 5.75) * 96; const maxH = (o.maxH || 7.6) * 96;
  let w = maxW; let h = (s.h / s.w) * w;
  if (h > maxH) { h = maxH; w = (s.w / s.h) * h; }
  return [
    new Paragraph({ children: [new ImageRun({ type: s.type, data: fs.readFileSync(file), transformation: { width: Math.round(w), height: Math.round(h) },
      altText: { title: label, description: caption, name: path.basename(file) } })],
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 } }),
    new Paragraph({ children: [new TextRun({ text: label + ' ', font: FONT, size: SZ, bold: true }), ...runs(caption)],
      alignment: AlignmentType.CENTER, spacing: { after: o.source ? 0 : 160 } }),
    ...(o.source ? [new Paragraph({ children: runs('ที่มา: ' + o.source, { size: 28 }), alignment: AlignmentType.CENTER, spacing: { after: 160 } })] : []),
  ];
}
// บรรทัดสารบัญแบบมีจุดนำ
function tocLine(text, page, level = 0, bold = false) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: SZ, bold }), new TextRun({ text: '\t' + page, font: FONT, size: SZ, bold })],
    tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W, leader: 'dot' }],
    indent: { left: level * 500 + (level ? 0 : 0), hanging: 0 }, spacing: { after: 0 },
  });
}
function ref(text) {
  return new Paragraph({ children: runs(text), indent: { left: 720, hanging: 720 }, spacing: { after: 80 }, alignment: AlignmentType.LEFT });
}
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

module.exports = { d, FONT, SZ, TEXT_W, JUST, runs, P, C, blank, chapter, front, H2, H3, items, eq, table, figure, tocLine, ref, pageBreak, state, cell };
