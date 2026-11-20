fn main() {
    for f in ["IBM Plex Mono", "sans-serif"] {
        let mut t = imagine_core::vector::TextSpec::new("", 0.0, 0.0, 90.0, [1.0; 4]);
        t.font = f.into();
        t.weight = 100.0;
        let s = imagine_vector::text_range(&t, 0, 0);
        println!(
            "{f}: {:?} box {:?}",
            imagine_vector::text_geometry(&t, &s),
            imagine_vector::text_box_size(&t)
        );
    }
}
