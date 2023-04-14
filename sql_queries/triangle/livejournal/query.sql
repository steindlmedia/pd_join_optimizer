-- out(x, y, z) :- livejournal(x, y), livejournal(y, z), livejournal(z, x).
SELECT CAST(a.x AS BIGINT), CAST(a.y AS BIGINT), CAST(b.y AS BIGINT) AS z
FROM livejournal a
JOIN livejournal b ON a.y = b.x
JOIN livejournal c ON b.y = c.x
WHERE a.x = c.y;