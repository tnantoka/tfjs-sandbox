import type { NextPage } from 'next';
import Link from 'next/link';

const examples = ['hello'];

const Home: NextPage = () => {
  return (
    <>
      <h1>
        My Sandbox for{' '}
        <a
          href="https://www.tensorflow.org/js"
          target="_blank"
          rel="noopener noreferrer"
        >
          TensorFlow.js
        </a>
      </h1>

      <ul>
        {examples.map((example) => (
          <li key={example}>
            <Link href={`/${example}`}>{example}</Link>
            <br />
          </li>
        ))}
      </ul>
    </>
  );
};

export default Home;
