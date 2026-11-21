//! Structured data inside Photoshop files: action descriptors (used by
//! newer adjustment layers and text) and the PostScript-like "EngineData"
//! of text layers. Reading both, writing descriptors.

use anyhow::{Result, bail, ensure};

/// A descriptor value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Desc(Desc),
    List(Vec<Value>),
    Double(f64),
    /// Unit float: unit key ("#Prc", "#Pnt" …) and value.
    Unit(String, f64),
    Text(String),
    Enum(String, String),
    Long(i32),
    Big(i64),
    Bool(bool),
    Raw(Vec<u8>),
    Class(String),
}

/// An action descriptor: class and ordered items.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Desc {
    pub class: String,
    pub items: Vec<(String, Value)>,
}

impl Desc {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn num(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            Value::Double(v) | Value::Unit(_, v) => Some(*v),
            Value::Long(v) => Some(*v as f64),
            Value::Big(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn text(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            Value::Text(t) => Some(t),
            _ => None,
        }
    }
    pub fn desc(&self, key: &str) -> Option<&Desc> {
        match self.get(key)? {
            Value::Desc(d) => Some(d),
            _ => None,
        }
    }
    pub fn raw(&self, key: &str) -> Option<&[u8]> {
        match self.get(key)? {
            Value::Raw(r) => Some(r),
            _ => None,
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "descriptor truncated");
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.bytes(4)?.try_into()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_be_bytes(self.bytes(8)?.try_into()?))
    }
    fn unicode(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        let units: Vec<u16> = self
            .bytes(n * 2)?
            .chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        Ok(String::from_utf16_lossy(&units)
            .trim_end_matches('\0')
            .to_string())
    }
    /// A key: length-prefixed string, or a 4-byte id when the length is 0.
    fn key(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        let n = if n == 0 { 4 } else { n };
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }
    fn desc(&mut self) -> Result<Desc> {
        let _name = self.unicode()?;
        let class = self.key()?;
        let count = self.u32()? as usize;
        let mut items = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let k = self.key()?;
            let v = self.value()?;
            items.push((k, v));
        }
        Ok(Desc { class, items })
    }
    fn value(&mut self) -> Result<Value> {
        let t: [u8; 4] = self.bytes(4)?.try_into()?;
        Ok(match &t {
            b"Objc" | b"GlbO" => Value::Desc(self.desc()?),
            b"VlLs" => {
                let n = self.u32()? as usize;
                Value::List((0..n).map(|_| self.value()).collect::<Result<_>>()?)
            }
            b"doub" => Value::Double(self.f64()?),
            b"UntF" => {
                let unit = String::from_utf8_lossy(self.bytes(4)?).into_owned();
                Value::Unit(unit, self.f64()?)
            }
            b"UnFl" => {
                let unit = String::from_utf8_lossy(self.bytes(4)?).into_owned();
                let n = self.u32()? as usize;
                let v: Vec<Value> = (0..n)
                    .map(|_| self.f64().map(|f| Value::Unit(unit.clone(), f)))
                    .collect::<Result<_>>()?;
                Value::List(v)
            }
            b"TEXT" => Value::Text(self.unicode()?),
            b"enum" => {
                let ty = self.key()?;
                Value::Enum(ty, self.key()?)
            }
            b"long" => Value::Long(self.u32()? as i32),
            b"comp" => Value::Big(i64::from_be_bytes(self.bytes(8)?.try_into()?)),
            b"bool" => Value::Bool(self.bytes(1)?[0] != 0),
            b"type" | b"GlbC" => {
                let _ = self.unicode()?;
                Value::Class(self.key()?)
            }
            b"alis" | b"tdta" | b"Pth " => {
                let n = self.u32()? as usize;
                Value::Raw(self.bytes(n)?.to_vec())
            }
            b"obj " => {
                // References: skip their items (not needed here).
                let n = self.u32()? as usize;
                for _ in 0..n {
                    let kind: [u8; 4] = self.bytes(4)?.try_into()?;
                    match &kind {
                        b"prop" => {
                            self.unicode()?;
                            self.key()?;
                            self.key()?;
                        }
                        b"Clss" => {
                            self.unicode()?;
                            self.key()?;
                        }
                        b"Enmr" => {
                            self.unicode()?;
                            self.key()?;
                            self.key()?;
                            self.key()?;
                        }
                        b"rele" => {
                            self.unicode()?;
                            self.key()?;
                            self.u32()?;
                        }
                        b"Idnt" | b"indx" => {
                            self.u32()?;
                        }
                        b"name" => {
                            self.unicode()?;
                            self.key()?;
                            self.unicode()?;
                        }
                        _ => bail!("unknown reference kind {kind:?}"),
                    }
                }
                Value::List(Vec::new())
            }
            _ => bail!("unknown descriptor type {:?}", String::from_utf8_lossy(&t)),
        })
    }
}

/// Reads a descriptor (after its 4-byte version field).
pub fn read_desc(b: &[u8]) -> Result<Desc> {
    R { b, p: 0 }.desc()
}

// --- Writing descriptors --------------------------------------------------

fn w_unicode(o: &mut Vec<u8>, s: &str) {
    let u: Vec<u16> = s.encode_utf16().chain(std::iter::once(0)).collect();
    o.extend((u.len() as u32).to_be_bytes());
    for c in u {
        o.extend(c.to_be_bytes());
    }
}

fn w_key(o: &mut Vec<u8>, k: &str) {
    if k.len() == 4 {
        o.extend(0u32.to_be_bytes());
    } else {
        o.extend((k.len() as u32).to_be_bytes());
    }
    o.extend(k.as_bytes());
}

fn w_value(o: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Desc(d) => {
            o.extend(b"Objc");
            w_desc(o, d);
        }
        Value::List(l) => {
            o.extend(b"VlLs");
            o.extend((l.len() as u32).to_be_bytes());
            for x in l {
                w_value(o, x);
            }
        }
        Value::Double(f) => {
            o.extend(b"doub");
            o.extend(f.to_be_bytes());
        }
        Value::Unit(u, f) => {
            o.extend(b"UntF");
            o.extend(&u.as_bytes()[..4]);
            o.extend(f.to_be_bytes());
        }
        Value::Text(t) => {
            o.extend(b"TEXT");
            w_unicode(o, t);
        }
        Value::Enum(t, e) => {
            o.extend(b"enum");
            w_key(o, t);
            w_key(o, e);
        }
        Value::Long(i) => {
            o.extend(b"long");
            o.extend(i.to_be_bytes());
        }
        Value::Big(i) => {
            o.extend(b"comp");
            o.extend(i.to_be_bytes());
        }
        Value::Bool(b) => {
            o.extend(b"bool");
            o.push(*b as u8);
        }
        Value::Raw(r) => {
            o.extend(b"tdta");
            o.extend((r.len() as u32).to_be_bytes());
            o.extend(r);
        }
        Value::Class(c) => {
            o.extend(b"type");
            w_unicode(o, "");
            w_key(o, c);
        }
    }
}

pub fn w_desc(o: &mut Vec<u8>, d: &Desc) {
    w_unicode(o, "");
    w_key(o, &d.class);
    o.extend((d.items.len() as u32).to_be_bytes());
    for (k, v) in &d.items {
        w_key(o, k);
        w_value(o, v);
    }
}

// --- EngineData -----------------------------------------------------------

/// A value of the text engine data.
#[derive(Debug, Clone, PartialEq)]
pub enum Ps {
    Dict(Vec<(String, Ps)>),
    Array(Vec<Ps>),
    Num(f64),
    /// A number written without a decimal point.
    Int(i64),
    Str(String),
    Name(String),
    Bool(bool),
}

impl Ps {
    pub fn get(&self, key: &str) -> Option<&Ps> {
        match self {
            Ps::Dict(d) => d.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Follows a path of keys and array indices ("EngineDict/StyleRun/RunArray/0").
    pub fn at(&self, path: &str) -> Option<&Ps> {
        let mut cur = self;
        for part in path.split('/') {
            cur = match (cur, part.parse::<usize>()) {
                (Ps::Array(a), Ok(i)) => a.get(i)?,
                _ => cur.get(part)?,
            };
        }
        Some(cur)
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Ps::Num(n) => Some(*n),
            Ps::Int(n) => Some(*n as f64),
            _ => None,
        }
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Ps> {
        match self {
            Ps::Dict(d) => d.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Like [`Ps::at`], for changing a value.
    pub fn at_mut(&mut self, path: &str) -> Option<&mut Ps> {
        let mut cur = self;
        for part in path.split('/') {
            cur = match part.parse::<usize>() {
                Ok(i) if matches!(cur, Ps::Array(_)) => match cur {
                    Ps::Array(a) => a.get_mut(i)?,
                    _ => unreachable!(),
                },
                _ => cur.get_mut(part)?,
            };
        }
        Some(cur)
    }
    /// Sets `key` in a dictionary (added at the end if missing).
    pub fn set(&mut self, key: &str, v: Ps) {
        if let Ps::Dict(d) = self {
            match d.iter_mut().find(|(k, _)| k == key) {
                Some(e) => e.1 = v,
                None => d.push((key.into(), v)),
            }
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Ps::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn array(&self) -> Option<&[Ps]> {
        match self {
            Ps::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// Parses EngineData.
pub fn engine_data(b: &[u8]) -> Result<Ps> {
    let mut p = 0;
    let v = ps_value(b, &mut p)?;
    Ok(v)
}

fn skip_ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && (b[*p].is_ascii_whitespace() || b[*p] == 0) {
        *p += 1;
    }
}

fn ps_value(b: &[u8], p: &mut usize) -> Result<Ps> {
    skip_ws(b, p);
    ensure!(*p < b.len(), "engine data truncated");
    match b[*p] {
        b'<' if b.get(*p + 1) == Some(&b'<') => {
            *p += 2;
            let mut items = Vec::new();
            loop {
                skip_ws(b, p);
                ensure!(*p < b.len(), "engine data truncated");
                if b[*p] == b'>' {
                    *p += 2;
                    return Ok(Ps::Dict(items));
                }
                ensure!(b[*p] == b'/', "engine data: key expected");
                let k = ps_name(b, p);
                let v = ps_value(b, p)?;
                items.push((k, v));
            }
        }
        b'[' => {
            *p += 1;
            let mut items = Vec::new();
            loop {
                skip_ws(b, p);
                ensure!(*p < b.len(), "engine data truncated");
                if b[*p] == b']' {
                    *p += 1;
                    return Ok(Ps::Array(items));
                }
                items.push(ps_value(b, p)?);
            }
        }
        b'(' => {
            *p += 1;
            let mut raw = Vec::new();
            while *p < b.len() && b[*p] != b')' {
                if b[*p] == b'\\' && *p + 1 < b.len() {
                    *p += 1;
                    raw.push(match b[*p] {
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        c => c,
                    });
                } else {
                    raw.push(b[*p]);
                }
                *p += 1;
            }
            *p += 1;
            let s = if raw.starts_with(&[0xfe, 0xff]) {
                let u: Vec<u16> = raw[2..]
                    .chunks(2)
                    .filter(|c| c.len() == 2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect();
                String::from_utf16_lossy(&u)
            } else {
                String::from_utf8_lossy(&raw).into_owned()
            };
            Ok(Ps::Str(s))
        }
        b'/' => Ok(Ps::Name(ps_name(b, p))),
        _ => {
            let start = *p;
            while *p < b.len() && !b[*p].is_ascii_whitespace() && !b"[]<>()/".contains(&b[*p]) {
                *p += 1;
            }
            let tok = std::str::from_utf8(&b[start..*p]).unwrap_or("");
            match tok {
                "true" => Ok(Ps::Bool(true)),
                "false" => Ok(Ps::Bool(false)),
                t if !t.contains(['.', 'e', 'E']) && t.parse::<i64>().is_ok() => Ok(Ps::Int(t.parse().unwrap())),
                t => t
                    .parse::<f64>()
                    .map(Ps::Num)
                    .map_err(|_| anyhow::anyhow!("engine data: unexpected {t:?}")),
            }
        }
    }
}

/// Writes EngineData the way Photoshop does (tab-indented, UTF-16 strings).
pub fn write_engine_data(v: &Ps) -> Vec<u8> {
    fn num(o: &mut Vec<u8>, n: f64) {
        // Photoshop: "1.0", ".5", "-.25"
        let mut s = format!("{n:?}");
        if s.contains('e') {
            s = format!("{n:.6}");
        }
        // (but zero stays "0.0").
        if let Some(r) = s.strip_prefix("0.").filter(|r| *r != "0") {
            s = format!(".{r}");
        } else if let Some(r) = s.strip_prefix("-0.").filter(|r| *r != "0") {
            s = format!("-.{r}");
        }
        o.extend(s.as_bytes());
    }
    fn value(o: &mut Vec<u8>, v: &Ps, depth: usize) {
        let tabs = |o: &mut Vec<u8>, n: usize| o.extend(std::iter::repeat_n(b'\t', n));
        match v {
            Ps::Dict(items) => {
                o.extend(b"<<\n");
                for (k, v) in items {
                    tabs(o, depth + 1);
                    o.push(b'/');
                    o.extend(k.as_bytes());
                    // Nested dictionaries and arrays of them start on their own line.
                    let block = matches!(v, Ps::Dict(_)) || matches!(v, Ps::Array(a) if a.iter().any(|x| matches!(x, Ps::Dict(_))));
                    if block && matches!(v, Ps::Dict(_)) {
                        o.push(b'\n');
                        tabs(o, depth + 1);
                    } else {
                        o.push(b' ');
                    }
                    value(o, v, depth + 1);
                    o.push(b'\n');
                }
                tabs(o, depth);
                o.extend(b">>");
            }
            Ps::Array(items) => {
                if items.iter().any(|x| matches!(x, Ps::Dict(_))) {
                    o.extend(b"[");
                    for x in items {
                        o.push(b'\n');
                        tabs(o, depth);
                        value(o, x, depth);
                    }
                    o.push(b'\n');
                    tabs(o, depth);
                    o.push(b']');
                } else {
                    o.extend(b"[");
                    for x in items {
                        o.push(b' ');
                        value(o, x, depth);
                    }
                    o.extend(b" ]");
                }
            }
            Ps::Num(n) => num(o, *n),
            Ps::Int(n) => o.extend(n.to_string().as_bytes()),
            Ps::Bool(b) => o.extend(if *b { &b"true"[..] } else { &b"false"[..] }),
            Ps::Name(n) => {
                o.push(b'/');
                o.extend(n.as_bytes());
            }
            Ps::Str(s) => {
                o.extend(b"(\xfe\xff");
                for u in s.encode_utf16() {
                    for byte in u.to_be_bytes() {
                        if matches!(byte, b'(' | b')' | b'\\') {
                            o.push(b'\\');
                        }
                        o.push(byte);
                    }
                }
                o.push(b')');
            }
        }
    }
    let mut o = b"\n\n".to_vec();
    value(&mut o, v, 0);
    o
}

fn ps_name(b: &[u8], p: &mut usize) -> String {
    *p += 1;
    let start = *p;
    while *p < b.len() && !b[*p].is_ascii_whitespace() && !b"[]<>()/".contains(&b[*p]) {
        *p += 1;
    }
    String::from_utf8_lossy(&b[start..*p]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_roundtrip() {
        let d = Desc {
            class: "null".into(),
            items: vec![
                ("Brgh".into(), Value::Long(-20)),
                ("Cntr".into(), Value::Long(35)),
                ("useLegacy".into(), Value::Bool(false)),
                ("Nm  ".into(), Value::Text("Grüße".into())),
                ("Strt".into(), Value::Unit("#Prc".into(), 12.5)),
            ],
        };
        let mut b = Vec::new();
        w_desc(&mut b, &d);
        let back = read_desc(&b).unwrap();
        assert_eq!(back.num("Brgh"), Some(-20.0));
        assert_eq!(back.bool("useLegacy"), Some(false));
        assert_eq!(back.text("Nm  "), Some("Grüße"));
        assert_eq!(back.num("Strt"), Some(12.5));
    }

    #[test]
    fn engine_data_parses_dicts_arrays_and_utf16() {
        let src = b"<< /EngineDict << /Editor << /Text (\xfe\xff\x00H\x00i\x00\\\r) >> /StyleRun << /RunArray [ << /StyleSheet << /StyleSheetData << /FontSize 48.0 /FauxBold false >> >> >> ] >> >> >>";
        let v = engine_data(src).unwrap();
        assert_eq!(
            v.at("EngineDict/Editor/Text").and_then(Ps::str),
            Some("Hi\r")
        );
        assert_eq!(
            v.at("EngineDict/StyleRun/RunArray/0/StyleSheet/StyleSheetData/FontSize")
                .and_then(Ps::num),
            Some(48.0)
        );
    }
}

#[cfg(test)]
mod engine_writer_tests {
    use super::*;

    #[test]
    fn photoshop_engine_data_is_written_back_byte_for_byte() {
        let src: &[u8] = include_bytes!("psd_engine_template.dat");
        let v = engine_data(src).unwrap();
        let out = write_engine_data(&v);
        assert_eq!(engine_data(&out).unwrap(), v, "same tree");
        if out != src {
            let i = out.iter().zip(src).position(|(a, b)| a != b).unwrap_or(out.len().min(src.len()));
            panic!(
                "differs at {i}:\n ours: {:?}\n  ps:  {:?}",
                String::from_utf8_lossy(&out[i.saturating_sub(40)..(i + 40).min(out.len())]),
                String::from_utf8_lossy(&src[i.saturating_sub(40)..(i + 40).min(src.len())])
            );
        }
    }
}
