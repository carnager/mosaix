fn main() {
    let q = std::env::args().nth(1).unwrap_or_default().to_lowercase();
    let t = std::time::Instant::now();
    let g = imagine_vector::font_groups();
    println!("{} groups in {:?}", g.len(), t.elapsed());
    for k in imagine_vector::FontKind::ALL { println!("{:?}: {} (latin {})", k, g.iter().filter(|x| x.kind == k).count(), g.iter().filter(|x| x.kind == k && x.latin).count()); }
    for gr in g.iter().filter(|g| !q.is_empty() && g.name.to_lowercase().contains(&q)) {
        println!("{} {:?} ({} faces)", gr.name, gr.kind, gr.faces.len());
    }
}
