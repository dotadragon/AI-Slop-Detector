import { useEffect, useRef, useState } from "react";

export function useHumanSignal() {
  const startedAt = useRef(Date.now());
  const [moves, setMoves] = useState(0);
  const [keys, setKeys] = useState(0);

  useEffect(() => {
    const mouse = () => setMoves((value) => value + 1);
    const key = () => setKeys((value) => value + 1);
    window.addEventListener("mousemove", mouse);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousemove", mouse);
      window.removeEventListener("keydown", key);
    };
  }, []);

  return moves > 3 || keys > 0 || Date.now() - startedAt.current > 2000;
}
