fn main() {
    let q = std::env::args().nth(1).unwrap_or_default().to_lowercase();
    let g = imagine_vector::font_groups();
    println!("{} groups (from {} families)", g.len(), imagine_vector::font_families().len());
    for gr in g.iter().filter(|g| g.name.to_lowercase().contains(&q)) {
        println!("{}: {}", gr.name, gr.faces.iter().map(|f| format!("{} [{}]", f.label, f.family)).collect::<Vec<_>>().join(", "));
    }
}
