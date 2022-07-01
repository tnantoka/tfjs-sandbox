import '../styles/globals.css';
import type { AppProps } from 'next/app';
import Head from 'next/head';
import Link from 'next/link';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>tfjs-sandbox</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component {...pageProps} />
      <footer>
        <Link href="/">tfjs-sandbox</Link> by{' '}
        <a
          href="https://twitter.com/tnantoka"
          target="_blank"
          rel="noopener noreferrer"
        >
          @tnantoka
        </a>
      </footer>
    </>
  );
}

export default MyApp;
