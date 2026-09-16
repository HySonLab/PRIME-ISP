import argparse
from pymol import cmd
import os
from glob import glob
from tqdm import tqdm

import trimesh
from utils.partition import mesh_simplification_quadric_decimation
import subprocess

MAX_FACES = 1024

def export_surface(
    pdb_path,
    output_path,
    surface_quality=0,
    solvent_radius=1.4,
    selection="all",
):
    script = (
        "import pymol\n"
        "pymol.finish_launching(['pymol', '-qc'])\n"
        "from pymol import cmd\n"
        f"cmd.load('{pdb_path}', 'prot')\n"
        f"cmd.hide('everything', '{selection}')\n"
        f"cmd.show('surface', '{selection}')\n"
        f"cmd.set('surface_quality', {surface_quality})\n"
        f"cmd.set('solvent_radius', {solvent_radius})\n"
        f"cmd.save('{output_path}', '{selection}')\n"
        "cmd.delete('all')\n"
    )

    script_path = output_path.replace(".obj", "_pymol_script.py")
    with open(script_path, "w") as f:
        f.write(script)

    try:
        result = subprocess.run(
            ["python", script_path],
            timeout=60,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise ValueError(f"PyMol failed: {result.stderr}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(f"Surface file empty or missing: {output_path}")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def main():
    parser = argparse.ArgumentParser(
        description="Export protein surface(s) (OBJ) from PDB using PyMOL"
    )

    # --------------------------------------------------
    # Either single PDB OR directory
    # --------------------------------------------------
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--pdb",
        help="Path to single PDB file",
    )

    group.add_argument(
        "--pdb_dir",
        help="Directory containing multiple PDB files",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output Ply path (for single PDB) OR output directory (for pdb_dir mode)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Surface quality: 0=fast, 1=default, 2=high",
    )

    parser.add_argument(
        "--probe",
        type=float,
        default=1.4,
        help="Solvent probe radius (Å)",
    )

    parser.add_argument(
        "--selection",
        default="all",
        help="PyMOL selection (e.g. 'chain A')",
    )

    args = parser.parse_args()

    # ==================================================
    # Single file mode
    # ==================================================
    if args.pdb:

        print(f"Exporting surface for {args.pdb}")

        export_surface(
            pdb_path=args.pdb,
            output_path=args.out,
            surface_quality=args.quality,
            solvent_radius=args.probe,
            selection=args.selection,
        )
        
        mesh = trimesh.load(args.out)
        mesh = mesh_simplification_quadric_decimation(
            mesh,
            target_faces=MAX_FACES
        )
        mesh.export(args.out)

    # ==================================================
    # Directory mode
    # ==================================================
    else:

        os.makedirs(args.out, exist_ok=True)

        pdb_files = sorted(glob(os.path.join(args.pdb_dir, "*.pdb")))
        print(f"Found {len(pdb_files)} PDB files")

        for pdb_path in tqdm(pdb_files, desc="Exporting surfaces"):
            
            pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
            tmp_path = os.path.join(args.out, f"tmp.obj")
            out_path = os.path.join(args.out, f"{pdb_name}.ply")

            print(f"Exporting {pdb_name}")

            export_surface(
                pdb_path=pdb_path,
                output_path=tmp_path,
                surface_quality=args.quality,
                solvent_radius=args.probe,
                selection=args.selection,
            )
            
            mesh = trimesh.load(tmp_path, process=False)
            mesh = mesh_simplification_quadric_decimation(
                mesh,
                target_faces=MAX_FACES
            )
            mesh.vertex_normals = None
            mesh.face_normals = None
            mesh.export(out_path)

    cmd.quit()

if __name__ == "__main__":
    main()