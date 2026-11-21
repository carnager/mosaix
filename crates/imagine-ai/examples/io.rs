fn main() -> anyhow::Result<()> {
    let env = ort::init_from("/usr/lib/libonnxruntime.so")?.build()?;
    for f in std::env::args().skip(1) {
        let s = ort::session::Session::builder(&env)?.commit_from_file(&f)?;
        println!("== {f}");
        for i in s.inputs() {
            println!("  in  {} {:?}", i.name(), i.dtype());
        }
        for o in s.outputs() {
            println!("  out {} {:?}", o.name(), o.dtype());
        }
    }
    Ok(())
}
