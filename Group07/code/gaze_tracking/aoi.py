from dataclasses import dataclass


@dataclass(frozen=True)
class AOI:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


def build_grid_aois(width: int, height: int, rows: int = 3, cols: int = 3) -> list[AOI]:
    aois: list[AOI] = []
    cell_w = width / cols
    cell_h = height / rows

    for row in range(rows):
        for col in range(cols):
            aois.append(
                AOI(
                    id=f"R{row + 1}C{col + 1}",
                    x1=int(col * cell_w),
                    y1=int(row * cell_h),
                    x2=int((col + 1) * cell_w),
                    y2=int((row + 1) * cell_h),
                )
            )
    return aois


def find_aoi(x: float, y: float, aois: list[AOI]) -> str:
    for aoi in aois:
        if aoi.contains(x, y):
            return aoi.id
    return "OUT"
