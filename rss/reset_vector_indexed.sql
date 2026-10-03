-- Reset vector_indexed flag for all records in rss_articles
-- This will cause all articles to be re-indexed on the next sync run

UPDATE rss_articles
SET vector_indexed = FALSE;

-- Verify the update
SELECT COUNT(*) AS total_records,
       COUNT(*) FILTER (WHERE vector_indexed = FALSE) AS reset_count
FROM rss_articles;
